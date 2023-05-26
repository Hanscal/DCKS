# -*- coding: utf-8 -*-

"""
@Time    : 2023/5/24 7:00 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import json
import os
import timeit
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score
from transformers import BartConfig, AutoTokenizer

from src.utils.metrics import dialogue_evaluation
from src.dataset.loader import OurDataset, OurDataCollator
from src.model.our_model import BartForConditionalGeneration
from config import MAP_EMO, args
import logging as logger


def print_bleu_rouge(results):
    for k, v in results.items():
        print(f"***** {k}: {v} *****")

def evaluate(args, model: BartForConditionalGeneration, tokenizer, prefix=""):
    dataset = OurDataset(
        tokenizer,
        type_path=args.eval_prefix,
        data_dir=args.data_dir,
        n_obs=-1,
        max_target_length=args.max_target_length,
        max_source_length=args.max_source_length,
        prefix=model.config.prefix or "",
    )
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    golds = dataset.read_targets()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 collate_fn=OurDataCollator(tokenizer, args))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    src_texts = []
    emos = []
    ppls = []
    acc = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        kno_input_ids = batch["kno_input_ids"].to(args.device)
        kno_texts = batch['kno_texts'] # 这里只是为了比较
        kno_attention_mask = batch["kno_attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        emo_labels = batch["emo_labels"].to(args.device)
        inputs = {
            "input_ids": (input_ids, kno_input_ids),
            "attention_mask": (attention_mask, kno_attention_mask),
            "labels": labels,
            "emo_labels": emo_labels,
            "return_dict": True,
            "reduction": "none",
        }
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            mask = labels == tokenizer.pad_token_id
            # (bsz, len)
            res = model(**inputs)
            mlm_loss = res['mlm_loss']
            emo_loss = res['emo_loss']
            loss = mlm_loss + emo_loss if emo_loss else mlm_loss
            loss = loss.view(batch_size, -1).masked_fill(mask, 0)
            mlm_loss = mlm_loss.view(batch_size, -1).masked_fill(mask, 0)
            print("  loss: {:.4f}  mlm_loss:{:.2f} emo_loss:{:.2f}".format(torch.mean(loss), torch.mean(mlm_loss), torch.mean(emo_loss)))
            ppl = (mlm_loss.sum(dim=1) / (1 - mask.float()).sum(dim=1)).exp()
            ppls.extend(ppl.tolist())
            # 求emo的准确率
            emo_logits = res['emo_logits']
            if emo_labels.dim() == 2:
                emo_labels = emo_labels.squeeze(1).detach().cpu().numpy()
            pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)

            kno_texts_batch = [kno_texts[i * args.max_num_kno:(i + 1) * args.max_num_kno] for i in range(args.eval_batch_size)]
            emo_know = [(MAP_EMO[i],j) for i,j in zip(pred_program,kno_texts_batch)]
            emos.extend(emo_know)
            program_acc = accuracy_score(emo_labels, pred_program)
            acc.append(program_acc)
        seqs = model.generate(
            input_ids=(input_ids, kno_input_ids), max_length=args.max_target_length, use_cache=True,
            attention_mask=(attention_mask, kno_attention_mask), num_beams=args.num_beams,
            do_sample=args.do_sample, early_stopping=True,
            top_k=args.top_k,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
        )
        seqs = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        preds.extend(seqs)
        src_texts.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    evalTime = timeit.default_timer() - start_time
    assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}."
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    results = dialogue_evaluation(preds, golds)
    dist1, dist2 = results['dist1'], results['dist2']

    if acc:
        acc = np.mean(acc)
        print("ppl: {:.2f}, dist1: {:.2f}, dist2: {:.2f}, emo_acc: {:.2f}".format(sum(ppls) / len(ppls), dist1, dist2, acc))
        results['emo_acc'] = acc
    else:
        print("ppl: {:.2f}, dist1: {:.2f}, dist2: {:.2f}".format(sum(ppls) / len(ppls), dist1, dist2))
    results['ppl'] = sum(ppls) / len(ppls)

    return results, (preds, golds, emos, src_texts)

if __name__ == '__main__':
    # 利用预训练模型进行预测
    config: BartConfig = BartConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = BartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    results, (eval_preds, eval_golds, emos, src_texts) = evaluate(args, model, tokenizer)
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    output_prediction_file = outdir / "predictions_{}_{}.txt".format(args.task, args.eval_prefix)
    output_result_file = outdir / "results_{}_{}.json".format(args.task, args.eval_prefix)
    logger.info(f" Writing predictions into {str(output_prediction_file)}")
    logger.info(f" Writing results into {str(output_result_file)}")
    with output_prediction_file.open('w') as f:
        for i, l1 in enumerate(eval_preds):
            f.write("cxt" + str(i) + ":" + str(src_texts[i]) + "\n")
            f.write("emos" + str(i) + ":" + str(emos[i]) + "\n")
            f.write("pred" + str(i) + ":" + l1.strip() + "\n")
            f.write("gold" + str(i) + ":" + eval_golds[i].strip() + "\n")

    with output_result_file.open('w') as f:
        json.dump(results, f, indent=4)

    ppl, acc = results['ppl'], results.get('emo_acc',None)
    dist_1, dist_2 = results['dist1'], results['dist2']
    bleu1, bleu2, bleu3, bleu4 = results['bleu1'], results['bleu2'], results['bleu3'], results['bleu4']
    rouge1, rouge2, rougel = results['rouge1'], results['rouge2'], results['rougeL']
    print("finish evaluate!")
    print_bleu_rouge(results)
