# import language_evaluation

import re
import math
import sys
import numpy as np
import six
import random
from typing import List
from six.moves import xrange  # pylint: disable=redefined-builtin

import more_itertools
from nltk.stem import porter
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from collections import defaultdict, Counter, namedtuple


def mean(lst):
    return sum(lst) / len(lst)

def _calc_ngram_dict(tokens:List[str], ngram:int, dict_ref=None):
    ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
    total = len(tokens)
    for i in range(0, total - ngram + 1):
        item = tuple(tokens[i:i + ngram])
        ngram_dict[item] += 1
    return ngram_dict

def _calc_cover(cand, gold, ngram):
    cand_dict = _calc_ngram_dict(cand, ngram)
    gold_dict = _calc_ngram_dict(gold, ngram)
    cover = 0
    total = 0
    for token, freq in cand_dict.items():
        if token in gold_dict:
            cover += min(freq, gold_dict[token])
        total += freq
    return cover, total

def _calc_cover_rate(cands, golds, ngram):
    """
    calc_cover_rate
    """
    cover = 0.0
    total = 0.000001
    for cand_tokens, gold_tokens in zip(cands, golds):
        cur_cover, cur_total = _calc_cover(cand_tokens, gold_tokens, ngram)
        cover += cur_cover
        total += cur_total
    return cover / total

def _calc_bp(cands, golds):
    c_count = 0.000001
    r_count = 0.0
    for cand_tokens, gold_tokens in zip(cands, golds):
        c_count += len(cand_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp

def calc_corpus_bleu(cands, golds):
    bp = _calc_bp(cands, golds)
    cover_rate1 = _calc_cover_rate(cands, golds, 1)
    cover_rate2 = _calc_cover_rate(cands, golds, 2)
    cover_rate3 = _calc_cover_rate(cands, golds, 3)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    return bleu1, bleu2, bleu3


def calc_sentence_bleu(cands, golds):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    sf = SmoothingFunction().method7
    for hyp, ref in zip(cands, golds):
        try:
            b1 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[1, 0, 0, 0])
        except ZeroDivisionError:
            b1 = 0.0
        try:
            b2 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.5, 0.5, 0, 0])
        except ZeroDivisionError:
            b2 = 0.0
        try:
            b3 = sentence_bleu([ref], hyp, smoothing_function=sf, weights=[0.34, 0.33, 0.33, 0])
        except ZeroDivisionError:
            b3 = 0.0
        bleu1.append(b1)
        bleu2.append(b2)
        bleu3.append(b3)
    return mean(bleu1), mean(bleu2), mean(bleu3)

def calc_corpus_bleu_new(hypothesis, references):
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    references = [[gold] for gold in references]
    sf = SmoothingFunction(epsilon=1e-12).method1
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
    return b1, b2, b3, b4

def _calc_distinct_ngram(cands, ngram):
    ngram_total = 0.00001
    ngram_distinct_count = 0.00001
    pred_dict = defaultdict(int)
    for cand_tokens in cands:
        _calc_ngram_dict(cand_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total

def _calc_sent_distinct_ngram(cand, ngram):
    ngram_total = 0.0000000001
    ngram_distinct_count = 0.0
    ngram_dict = defaultdict(int)
    for i in range(0, len(cand) - ngram + 1):
        item = tuple(cand[i:i + ngram])
        ngram_dict[item] += 1
    for _, freq in ngram_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total

def calc_corpus_distinct(cands):
    distinct1 = _calc_distinct_ngram(cands, 1)
    distinct2 = _calc_distinct_ngram(cands, 2)
    return distinct1, distinct2

def calc_sentence_distinct(cands):
    distinct1 = mean([_calc_sent_distinct_ngram(c, 1) for c in cands])
    distinct2 = mean([_calc_sent_distinct_ngram(c, 2) for c in cands])
    return distinct1, distinct2

def calc_corpus_f1(cands, golds):
    golden_word_total = 0.00000001
    pred_word_total = 0.00000001
    hit_word_total = 0.00000001
    for response, golden_response in zip(cands, golds):
        common = Counter(response) & Counter(golden_response)
        hit_word_total += sum(common.values())
        golden_word_total += len(golden_response)
        pred_word_total += len(response)
    p = hit_word_total / pred_word_total
    r = hit_word_total / golden_word_total
    f1 = 2 * p * r / (p + r)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')


def dialogue_evaluation(ori_cands, ori_golds):
    assert len(ori_cands) == len(ori_golds), f"num cand: {len(ori_cands)}, num gold: {len(ori_golds)}"
    cands = []
    golds = []
    help_tokenize = lambda x: word_tokenize(x.lower())
    for cand, gold in zip(ori_cands, ori_golds):
        cands.append(help_tokenize(cand.lower()))
        golds.append(help_tokenize(gold.lower()))
    cbleu1, cbleu2, cbleu3, cbleu4 = calc_corpus_bleu_new(cands, golds)
    sbleu1, sbleu2, sbleu3 = calc_sentence_bleu(cands, golds)
    cdiv1, cdiv2 = calc_corpus_distinct(cands)
    sdiv1, sdiv2 = calc_sentence_distinct(cands)
    results1 = RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer).run_evaluation(cands, golds)
    rouge1, rouge2, rougel = results1['rouge1'], results1['rouge2'], results1['rougeL']
    cf1 = calc_corpus_f1(cands, golds)
    result = {
        'cf1': cf1,
        'bleu1': cbleu1,
        'bleu2': cbleu2,
        'bleu3': cbleu3,
        'bleu4': cbleu4,
        'dist1': cdiv1,
        'dist2': cdiv2,
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougel': rougel
    }
    # result.update(rouge_result)
    result = {k: round(100 * v, 6) for k, v in result.items()}
    return result

def file_dialogue_evaluation(cand_file, gold_file):
    print(f"cand file: {cand_file}, gold file: {gold_file}")
    cands = []
    golds = []
    with open(cand_file, 'r', encoding='utf-8') as f:
        for line in f:
            cands.append(line.strip())
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            golds.append(line.strip())
    results = dialogue_evaluation(cands, golds)
    print(results)


def print_custum(emotion, dial, ref, hyp_b, hyp_g, pred_emotions, comet_res):
    res = ""
    res += "Emotion: {}".format(emotion) + "\n"
    if pred_emotions:
        res += "Pred Emotions: {}".format(pred_emotions) + "\n"
    if comet_res:
        for k, v in comet_res.items():
            res += "{}:{}".format(k, v) + "\n"
    res += "Context:{}".format(dial) + "\n"
    if hyp_b:
        res += "Beam:{}".format(hyp_b) + "\n"
    res += "Greedy:{}".format(hyp_g) + "\n"
    res += "Ref:{}".format(ref) + "\n"
    res += "---------------------------------------------------------------" + "\n"

    return res


def tokenize(text, stemmer):
  """Tokenize input text into a list of tokens.
  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = re.sub(r"[^a-z0-9]+", " ", text)

  tokens = re.split(r"\s+", text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]

  return tokens

class RougeScorer(object):
  """Calculate rouges scores between two blobs of text.
  Sample usage:
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
  """

  def __init__(self, rouge_types, use_stemmer=False, tokenization_fn=None):
    """Initializes a new RougeScorer.
    Valid rouge types that can be computed are:
      rougen (e.g. rouge1, rouge2): n-gram based scoring.
      rougeL: Longest common subsequence based scoring.
    Args:
      rouge_types: A list of rouge types to calculate.
      use_stemmer: Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching. (Only available with default tokenizer)
      tokenization_fn: Function that take string as input, and list of tokens as return
    Returns:
      A dict mapping rouge types to Score tuples.
    """

    self.rouge_types = rouge_types
    self._stemmer = porter.PorterStemmer() if use_stemmer else None
    self._tokenization_fn = tokenization_fn

  def score(self, target, prediction):
    """Calculates rouge scores between the target and prediction.
    Args:
      target: Text containing the target (ground truth) text.
      prediction: Text containing the predicted text.
    Returns:
      A dict mapping each rouge type to a Score object.
    Raises:
      ValueError: If an invalid rouge type is encountered.
    """

    if self._tokenization_fn:
        target_tokens = self._tokenization_fn(target)
        prediction_tokens = self._tokenization_fn(prediction)
    else:
        target_tokens = tokenize(target, self._stemmer)
        prediction_tokens = tokenize(prediction, self._stemmer)
    result = {}

    for rouge_type in self.rouge_types:
      if rouge_type == "rougeL":
        # Rouge from longest common subsequences.
        scores = _score_lcs(target_tokens, prediction_tokens)
      elif re.match(r"rouge[0-9]$", rouge_type):
        # Rouge from n-grams.
        n = int(rouge_type[5:])
        if n <= 0:
          raise ValueError("rougen requires positive n: %s" % rouge_type)
        target_ngrams = _create_ngrams(target_tokens, n)
        prediction_ngrams = _create_ngrams(prediction_tokens, n)
        scores = _score_ngrams(target_ngrams, prediction_ngrams)
      else:
        raise ValueError("Invalid rouge type: %s" % rouge_type)
      result[rouge_type] = scores

    return result


def _create_ngrams(tokens, n):
  """Creates ngrams from the given list of tokens.
  Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.
  Returns:
    A dictionary mapping each bigram to the number of occurrences.
  """

  ngrams = Counter()
  for ngram in (tuple(tokens[i:i + n]) for i in xrange(len(tokens) - n + 1)):
    ngrams[ngram] += 1
  return ngrams

class Score(namedtuple("Score", ["precision", "recall", "fmeasure"])):
  """Tuple containing precision, recall, and f-measure values."""


def fmeasure(precision, recall):
  """Computes f-measure given precision and recall values."""

  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0

def _score_lcs(target_tokens, prediction_tokens):
  """Computes LCS (Longest Common Subsequence) rouge scores.
  Args:
    target_tokens: Tokens from the target text.
    prediction_tokens: Tokens from the predicted text.
  Returns:
    A Score object containing computed scores.
  """

  if not target_tokens or not prediction_tokens:
    return Score(precision=0, recall=0, fmeasure=0)

  # Compute length of LCS from the bottom up in a table (DP appproach).
  cols = len(prediction_tokens) + 1
  rows = len(target_tokens) + 1
  lcs_table = np.zeros((rows, cols))
  for i in xrange(1, rows):
    for j in xrange(1, cols):
      if target_tokens[i - 1] == prediction_tokens[j - 1]:
        lcs_table[i, j] = lcs_table[i - 1, j - 1] + 1
      else:
        lcs_table[i, j] = max(lcs_table[i - 1, j], lcs_table[i, j - 1])
  lcs_length = lcs_table[-1, -1]

  precision = lcs_length / len(prediction_tokens)
  recall = lcs_length / len(target_tokens)
  fmeasures = fmeasure(precision, recall)

  return Score(precision=precision, recall=recall, fmeasure=fmeasures)


def _score_ngrams(target_ngrams, prediction_ngrams):
  """Compute n-gram based rouge scores.
  Args:
    target_ngrams: A Counter object mapping each ngram to number of
      occurrences for the target text.
    prediction_ngrams: A Counter object mapping each ngram to number of
      occurrences for the prediction text.
  Returns:
    A Score object containing computed scores.
  """

  intersection_ngrams_count = 0
  for ngram in six.iterkeys(target_ngrams):
    intersection_ngrams_count += min(target_ngrams[ngram],
                                     prediction_ngrams[ngram])
  target_ngrams_count = sum(target_ngrams.values())
  prediction_ngrams_count = sum(prediction_ngrams.values())

  precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
  recall = intersection_ngrams_count / max(target_ngrams_count, 1)
  fmeasures = fmeasure(precision, recall)

  return Score(precision=precision, recall=recall, fmeasure=fmeasures)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')

class RougeEvaluator(object):
    """Calculate rouges scores two blobs of single-sentence text by using
    google's python rouge scripts.
    (If you wnat to get sentence-level ROUGE-L, use Rouge155Evaluator)
    Sample usage:
        evaluator = language_evaluation.RougeEvaluator(
            rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        results = evaluator.run_evaluation(
            ['i am a boy', 'she is a girl'],
            ['am i a boy ?', 'is she a girl ?'])
    """
    def __init__(self,
                 num_parallel_calls: int = 1,
                 rouge_types=["rouge1", "rouge2", "rougeL"],
                 use_stemmer=True,
                 tokenization_fn=None,
                 average=True):
        self._num_parallel_calls = num_parallel_calls
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self._tokenization_fn = tokenization_fn
        self.average = average

    def run_evaluation(self, predicts, answers):
        n_predicts = self._split_list(predicts, self._num_parallel_calls)
        n_answers = self._split_list(answers, self._num_parallel_calls)
        from multiprocessing import Pool
        p = Pool(self._num_parallel_calls)
        import time
        start = time.time()
        results = p.map(self._run_evaluation, zip(n_predicts, n_answers))
        p.close()
        p.join()
        end = time.time()
        print(f"Takes {end-start} seconds for rouge evaluation with \
              {self._num_parallel_calls} processes")

        # results = self._run_evaluation([predicts, answers])
        # Average results form processes
        averaged_result = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for result in results:
            for key, value in result.items():
                averaged_result[key].append(value)
        if self.average:
            for key, value in averaged_result.items():
                # TODO : Currently, we assume each process has same numver of
                # predict-answer pairs
                averaged_result[key] = sum(value) / len(value)

        return averaged_result

    def _run_evaluation(self, predicts_and_answers):
        predicts, answers = predicts_and_answers
        scorer = RougeScorer(self.rouge_types, self.use_stemmer, self._tokenization_fn)
        scores = {rouge_type: [] for rouge_type in self.rouge_types}
        for predict, answer in zip(predicts, answers):
            # TODO : support multi-reference
            score = scorer.score(answer, predict)
            for key, value in score.items():
                scores[key].append(value.fmeasure)

        # Averaging
        for key in scores.keys():
            if self.average:
                scores[key] = np.mean(np.array(scores[key]))
            else:
                scores[key] = np.array(scores[key])

        return scores

    def _split_list(self, in_list, num_splits):
        return [list(c) for c in more_itertools.divide(num_splits, in_list)]

class FleissKappa(object):
    def __init__(self, score_scale=5, sample_number=109, person_number=5):
        self.score = score_scale
        self.sample_number = sample_number
        self.person_number = person_number

    def read_excel(self, file_path):
        import pandas as pd
        # 读取全部表格，key为sheetname
        data = pd.read_excel(io=file_path, sheet_name=None)
        return data

    def form_testdata(self, excel_data):
        # 形成testData形式
        # 从excel_data中随机选择3个
        excel_data_new = {}
        random_samples = random.sample(range(1,6), 5)
        sample_list = ['person'+str(i) for i in random_samples]
        for key, value in excel_data.items():
            if key in sample_list:
                excel_data_new[key] = value
        excel_data = excel_data_new
        # 建立一个空的2维矩阵
        empty_ = np.zeros((self.sample_number, self.score))
        # 模型个数
        res_dict = {str(i):[] for i in range(1,6)}
        res_mean_score = {str(i):'' for i in range(1,6)}
        for i in range(1,6):
            coh = empty_.copy()
            emp = empty_.copy()
            inf = empty_.copy()
            con = empty_.copy()
            # 对于连贯性 （N，5）N为样本数量，5为分数范围（1-5）
            # 对于共情性
            # 对于信息量
            # 对于持续性
            for sheet, data in excel_data.items():
                # print(i, sheet)
                data = data.fillna('')
                j = 0
                for idx, item in data.iterrows():
                    model_name = item['model']
                    coh_value = item['Coherence']
                    emp_value = item['Empthy']
                    inf_value = item['Informativeness']
                    con_value = item['Continuity']
                    if model_name and str(model_name) in '12345' and int(model_name) == i:
                        if not coh_value or not emp_value or not inf_value or not con_value:
                            print(item)
                        coh[j][int(coh_value) - 1] += 1
                        emp[j][int(emp_value) - 1] += 1
                        inf[j][int(inf_value) - 1] += 1
                        con[j][int(con_value) - 1] += 1
                        j += 1

            res_dict[str(i)] = [coh, emp, inf, con]
            # 计算平均分数
            coh_mean = np.mean(coh[:,0])*1 + np.mean(coh[:,1])*2 + np.mean(coh[:,2])*3 + np.mean(coh[:,3])*4 + np.mean(coh[:,4])*5
            emp_mean = np.mean(emp[:,0])*1 + np.mean(emp[:,1])*2 + np.mean(emp[:,2])*3 + np.mean(emp[:,3])*4 + np.mean(emp[:,4])*5
            inf_mean = np.mean(inf[:,0])*1 + np.mean(inf[:,1])*2 + np.mean(inf[:,2])*3 + np.mean(inf[:,3])*4 + np.mean(inf[:,4])*5
            con_mean = np.mean(con[:,0])*1 + np.mean(con[:,1])*2 + np.mean(con[:,2])*3 + np.mean(con[:,3])*4 + np.mean(con[:,4])*5
            res_mean_score[str(i)] = [coh_mean/self.person_number, emp_mean/self.person_number, inf_mean/self.person_number, con_mean/self.person_number]


        return res_dict, res_mean_score

    def fleiss_kappa(self, testData, N, k, n): #testData表示要计算的数据，（N,k）表示矩阵的形状，说明数据是N行j列的，一共有n个标注人员
        dataMat = np.mat(testData, float)
        oneMat = np.ones((k, 1))
        sum = 0.0
        P0 = 0.0
        for i in range(N):
            temp = 0.0
            for j in range(k):
                sum += dataMat[i, j]
                temp += 1.0*dataMat[i, j]**2
            temp -= n
            temp /= (n-1)*n
            P0 += temp
        P0 = 1.0*P0/N
        ysum = np.sum(dataMat, axis=0)
        for i in range(k):
            ysum[0, i] = (ysum[0, i]/sum)**2
        Pe = ysum*oneMat*1.0
        ans = (P0-Pe)/(1-Pe)
        return ans[0, 0]

    def run_fleiss_kappa(self, file_path):
        excel_data = self.read_excel(file_path)
        res_dict, res_score = self.form_testdata(excel_data)
        final_res = {}
        for model, testdata_list in res_dict.items():
            kappa_list = []
            for testdata in testdata_list:
                kappa = self.fleiss_kappa(testdata, self.sample_number, self.score, self.person_number)
                kappa_list.append(kappa)
            final_res[model] = kappa_list
        return final_res, res_score

    def form_ab_testdata(self, excel_data):
        # 形成testData形式
        # 从excel_data中随机选择3个
        excel_data_new = {}
        random_samples = random.sample(range(1, 6), 5)
        sample_list = ['person' + str(i) for i in random_samples]
        for key, value in excel_data.items():
            if key in sample_list:
                excel_data_new[key] = value
        excel_data = excel_data_new
        # 建立一个空的2维矩阵
        empty_ = np.zeros((self.sample_number, 3)) # lose, tie, win分别对应分数为-1，0, 1
        # 模型个数
        res_dict_ab = {str(i): [] for i in ['5v1','5v2','5v3','5v4']}
        res_mean_score = {str(i): '' for i in ['5v1','5v2','5v3','5v4']}
        for i in ['5v1','5v2','5v3','5v4']:
            coh = empty_.copy()
            emp = empty_.copy()
            inf = empty_.copy()
            con = empty_.copy()
            # 对于连贯性 （N，5）N为样本数量，5为分数范围（1-5）
            # 对于共情性
            # 对于信息量
            # 对于持续性
            for sheet, data in excel_data.items():
                # print(i, sheet)
                data = data.fillna('')
                j = 0
                for idx, item in data.iterrows():
                    model_name = item['model_vs']
                    coh_value = item['Coherence_ab']
                    emp_value = item['Empthy_ab']
                    inf_value = item['Informativeness_ab']
                    con_value = item['Continuity_ab']
                    if model_name and str(model_name) == i:
                        if coh_value =='' or emp_value =='' or inf_value =='' or con_value=='':
                            print(item)
                        if j >= self.sample_number:
                            print(j)
                        coh_value = int(item['Coherence_ab'])
                        emp_value = int(item['Empthy_ab'])
                        inf_value = int(item['Informativeness_ab'])
                        con_value = int(item['Continuity_ab'])
                        if coh_value < 0:
                            coh[j][0] += 1
                        elif coh_value == 0:
                            coh[j][1] += 1
                        elif coh_value > 0:
                            coh[j][2] += 1
                        if emp_value < 0:
                            emp[j][0] += 1
                        elif emp_value == 0:
                            emp[j][1] += 1
                        elif emp_value > 0:
                            emp[j][2] += 1
                        if inf_value < 0:
                            inf[j][0] += 1
                        elif inf_value == 0:
                            inf[j][1] += 1
                        elif inf_value > 0:
                            inf[j][2] += 1
                        if con_value < 0:
                            con[j][0] += 1
                        elif con_value == 0:
                            con[j][1] += 1
                        elif con_value > 0:
                            con[j][2] += 1
                        j += 1

            res_dict_ab[str(i)] = [coh, emp, inf, con]
            # 计算平均分数
            number_sp = self.sample_number*self.person_number
            coh_mean = [np.sum(coh[:, 0]) / number_sp, np.sum(coh[:, 1]) / number_sp, np.sum(coh[:, 2])/ number_sp]
            emp_mean = [np.sum(emp[:, 0]) / number_sp, np.sum(emp[:, 1]) / number_sp, np.sum(emp[:, 2])/ number_sp]
            inf_mean = [np.sum(inf[:, 0]) / number_sp, np.sum(inf[:, 1]) / number_sp, np.sum(inf[:, 2])/ number_sp]
            con_mean = [np.sum(con[:, 0]) / number_sp, np.sum(con[:, 1]) / number_sp, np.sum(con[:, 2])/ number_sp]
            res_mean_score[str(i)] = [coh_mean, emp_mean, inf_mean, con_mean]

        return res_dict_ab, res_mean_score

    def run_ab_test(self, file_path):
        excel_data = self.read_excel(file_path)
        res_dict_ab, res_score = self.form_ab_testdata(excel_data)
        final_res = {}
        for model, testdata_list in res_dict_ab.items():
            kappa_list = []
            for testdata in testdata_list:
                kappa = self.fleiss_kappa(testdata, self.sample_number, 3, self.person_number)
                kappa_list.append(kappa)
            final_res[model] = kappa_list
        return final_res, res_score


if __name__ == '__main__':
    from pprint import pprint
    # predicts = ['i am a boy', 'she is a girl']
    # answers = ['am i a boy ?', 'is she a girl ?']
    # evaluator = RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
    # results = evaluator.run_evaluation(predicts, answers)
    # print(results)
    file_path = "/Volumes/work/personal/papers/华院计算/kgconv/manuscript/human_test/human_test_109_ab.xlsx"
    fk = FleissKappa()
    data,score = fk.run_fleiss_kappa(file_path)
    data1, score1 = fk.run_ab_test(file_path)

    cand_file = sys.argv[1]
    gold_file = sys.argv[2]
    file_dialogue_evaluation(cand_file, gold_file)

    pprint(data)
    pprint(score)
    pprint(data1)
    pprint(score1)
