# -*- coding: utf-8 -*-

"""
@Time    : 2023/5/24 6:53 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import sys
import torch
import glob
import logging
import random
import logging as logger

from transformers import WEIGHTS_NAME, BartConfig, AutoTokenizer
sys.path.append('.')
from config import args
from src.utils.common import set_seed, print_opts, get_parameter_number
from src.dataset.loader import OurDataset

from src.trainer import train
from src.evaluator import evaluate
from src.model.our_model import BartForConditionalGeneration

torch.set_printoptions(precision=4)

# Setup CUDA, GPU
os.environ['CUDA_VISIBLE_DEVICES']= os.getenv('CUDA_VISIBLE_DEVICES') if os.getenv('CUDA_VISIBLE_DEVICES') else '0'
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else 1

# Set seed
set_seed()

def main():
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train  and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    config: BartConfig = BartConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    logger.info("Training/evaluation parameters %s")
    print_opts(args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
            apex.amp.register_float_function(torch, 'sigmoid')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        model = BartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model.to(args.device)
        logger.info(f"#Params: {get_parameter_number(model)}")
        train_dataset = OurDataset(
            tokenizer,
            type_path=args.train_prefix,
            data_dir=args.data_dir,
            n_obs=-1,
            max_target_length=args.max_target_length,
            max_source_length=args.max_source_length,
            prefix=model.config.prefix or "",
        )
        # 对train_dataset随机选择1/2, 1/4, 1/8
        train_dataset.examples = random.sample(train_dataset.examples, int(len(train_dataset)*args.data_ratio))
        logger.info("use train examples: {}".format(len(train_dataset)))
        print("use train examples: {}".format(len(train_dataset)))
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # evaluating
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # if args.do_train:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.model_name_or_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = BartForConditionalGeneration.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)
            # Evaluate
            result, _ = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info(f"Results: {results}")
    return results


if __name__ == "__main__":
    main()
