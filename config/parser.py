# -*- coding: utf-8 -*-

"""
@Time    : 2023/5/24 7:03 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import argparse

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name_or_path",default='facebook/bart-base',type=str,help="Path to pretrained model or model identifier from huggingface.co/models bart-base or bart-large",)
parser.add_argument("--kno_mlm_model_path",default=None,type=str,help="Path to pretrained model or model identifier from huggingface.co/models",)
parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name",)
parser.add_argument("--output_dir",default='checkpoints/ed_ours',type=str,help="The output directory where the model checkpoints and predictions will be written.",)
parser.add_argument("--task",default='ed_full',type=str)

# Other parameters
parser.add_argument("--data_dir",default='data/',type=str,help="The input data dir. Should contain the .json files for the task.")
parser.add_argument("--train_prefix",default="train", type=str, help="The input data dir. Should contain the .json files for the task.")
parser.add_argument("--eval_prefix", default="test",type=str, help="The input data dir. Should contain the .json files for the task.")
parser.add_argument("--cache_dir", default="cached",type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
parser.add_argument("--max_source_length", default=256, type=int,help="The maximum total input sequence length after WordPiece tokenization. Sequences "
         + "longer than this will be truncated, and sequences shorter than this will be padded.",)
parser.add_argument("--max_target_length", default=64, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences "
         + "longer than this will be truncated, and sequences shorter than this will be padded.",)
parser.add_argument("--max_kno_length", default=256, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences "
         + "longer than this will be truncated, and sequences shorter than this will be padded.",)
parser.add_argument("--max_knowl_length", default=16, type=int, help="The maximum length for one relation",)
parser.add_argument("--max_num_kno", default=5, type=int, help="The konw number for one relation")
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=10, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=10, type=int, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--max_steps",default=-1,type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--do_sample",action="store_true",)
parser.add_argument("--no_repeat_ngram_size", default=None, type=int,)
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints",action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
parser.add_argument("--fix_encoder", action="store_true")
parser.add_argument("--init_kno_encoder", action="store_true", help="Whether not to use CUDA when available")

parser.add_argument("--overwrite_output_dir", default=True, action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus only support -1 and 0")
parser.add_argument("--fp16", default=False, action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

parser.add_argument("--top_k", default=None, type=int,)
parser.add_argument("--repetition_penalty",default=1.5,type=float,)
parser.add_argument("--length_penalty",default=None,type=float,)
parser.add_argument("--woDiv", default=True, action="store_true")

# 一些需要消融实验的参数
parser.add_argument("--num_beams",default=5,type=int, help="1 means gready decode")

parser.add_argument("--woKnowledge", default=False, action="store_true")
parser.add_argument("--woContext", default=False, action="store_true")
parser.add_argument("--data_ratio", default=1, type=float)
args = parser.parse_args()
