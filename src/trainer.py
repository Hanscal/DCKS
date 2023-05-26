# -*- coding: utf-8 -*-

"""
@Time    : 2022/6/3 11:18 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
from pathlib import Path
import json
from typing import List

from tqdm import tqdm, trange
from collections import Counter


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.dataset.loader import OurDataCollator
from src.model.our_model import BartForConditionalGeneration
from src.utils.common import set_seed
from src.evaluator import evaluate
from config import args as config_args
from config import LS_idx, RS_idx, PAD_idx

import logging as logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Added here for reproductibility
set_seed()

def grad_status(model: nn.Module):
    return (par.requires_grad for par in model.parameters())


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(list(map(int, model_grads)))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"


def train(args, train_dataset, model: BartForConditionalGeneration, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter(filename_suffix=args.task)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=OurDataCollator(tokenizer, args))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.fix_encoder:
        assert_all_frozen(model.model.encoder)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                    any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    criterion_div = nn.NLLLoss(ignore_index=PAD_idx, reduction="sum")
    if not config_args.woDiv:
        criterion_div.weight = torch.ones(tokenizer.vocab_size)
    word_freq = np.zeros(tokenizer.vocab_size)
    def clean_preds(preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if RS_idx in pred:
                ind = pred.index(RS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == LS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != RS_idx:
                word_freq[k] += v

    def calc_weight():
        RF = word_freq / word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(args.device)

    tr_loss, logging_loss, loss, bppl, bacc = 0.0, 0.0, np.inf, 1000, 0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Epoch-{}".format(epoch), disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            kno_input_ids = batch["kno_input_ids"].to(args.device)
            kno_attention_mask = batch["kno_attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            emo_labels = batch["emo_labels"].to(args.device)
            inputs = {
                "input_ids": (input_ids, kno_input_ids),
                "attention_mask": (attention_mask, kno_attention_mask),
                "labels": labels,
                "emo_labels": emo_labels,
                "return_dict": True,
                "reduction": "mean",
            }
            res = model(**inputs)
            mlm_loss = res['mlm_loss']
            emo_loss = res['emo_loss']
            # todo 消融实验 woDiv
            logits = res['logits'] # mlm logits
            if not (config_args.woDiv):
                _, preds = logits.max(dim=-1)
                preds = clean_preds(preds)
                update_frequency(preds)
                criterion_div.weight = calc_weight()
                not_pad = labels.ne(PAD_idx)
                target_tokens = not_pad.long().sum().item()
                div_loss = criterion_div(
                    logits.contiguous().view(-1, logits.size(-1)),
                    labels.contiguous().view(-1),  # 应该是解码出来字符的id
                )
                div_loss /= target_tokens
                loss = emo_loss + div_loss + mlm_loss
                if step % (args.logging_steps // 10) == 0:
                    print("  loss: {:.4f}  mlm_loss:{:.2f} emo_loss:{:.2f} div_loss:{:.2f}".format(loss, mlm_loss, emo_loss, div_loss))
            else:
                loss = emo_loss + mlm_loss
                if step % (args.logging_steps // 10) == 0:
                    print("  loss: {:.4f}  mlm_loss:{:.2f} emo_loss:{:.2f}".format(loss, mlm_loss, emo_loss))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            print()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if (args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0) or \
                        (args.save_steps > 0 and global_step % args.save_steps == 0):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, (eval_preds, eval_golds, emos, src_texts) = evaluate(args, model, tokenizer)
                        ppl = results.get('ppl', np.inf)
                        acc = results.get('emo_acc', 0)
                        # Save model checkpoint, 根据ppl来保存模型
                        if bppl > ppl:
                            bppl = ppl
                            output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}".format(args.task, global_step))
                            # Take care of distributed/parallel training
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

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
                        # save model checkpoint, 根据acc来保存模型
                        if acc > bacc:
                            bacc = acc
                            output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}-acc".format(args.task, global_step))
                            # todo 保存模型states和scheduler
                            # Take care of distributed/parallel training
                            # 仅保存模型和tokenizer
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            outdir = Path(args.output_dir)
                            outdir.mkdir(exist_ok=True, parents=True)
                            output_prediction_file = outdir / "predictions_{}_{}_acc.txt".format(args.task, args.eval_prefix)
                            output_result_file = outdir / "results_{}_{}_acc.json".format(args.task, args.eval_prefix)
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

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step
