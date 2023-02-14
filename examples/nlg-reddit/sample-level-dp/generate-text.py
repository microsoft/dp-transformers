#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import collections
import csv
import logging
import os.path
import random

import numpy as np
import torch

import transformers
from transformers import AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from dp_transformers.module_modification import convert_gpt2_attention_to_lora

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-medium": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-large": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-xl": (GPT2LMHeadModel, GPT2Tokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def cal_perplexity(encodings, cur_model):
    max_length = cur_model.config.n_positions
    stride = 512
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    nlls_cur = []

    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[target_ids==cur_model.config.pad_token_id] = -100

        with torch.no_grad():
            outputs = cur_model(input_ids, labels=target_ids)
            nlls_cur.append(outputs[0] * trg_len)

    ppl_cur = torch.exp(torch.stack(nlls_cur).sum() / end_loc)

    return ppl_cur.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--input_training_file",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--do_sample", action="store_true", help="sampling when generation")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--total_sequences", type=int, default=100000, help="The number of total samples to generate.")


    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument("--lora_dim", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    
    if args.lora_dim > 0:
        if tokenizer.pad_token_id:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_type, pad_token_id=tokenizer.pad_token_id)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_type, pad_token_id=tokenizer.eos_token_id)

        model.resize_token_embeddings(len(tokenizer))

        model = convert_gpt2_attention_to_lora(
            model, r=args.lora_dim, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )

        state_dict = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"), map_location="cpu")
        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = model_class._load_pretrained_model(
                model,
                state_dict,
                [k for k in state_dict.keys()],  # XXX: rename?
                os.path.join(args.model_name_or_path, "pytorch_model.bin"),
                args.model_name_or_path,
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()
    else:
        if tokenizer.pad_token_id:
            model = model_class.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.pad_token_id)
        else:
            model = model_class.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.eos_token_id)

    model.eval()
    model.to(args.device)

    if args.fp16:
        model.half()

    logger.info(args)

    def generate_text(prompt,seq_num,prompt_length):
        ppls_cur = []
        all_data = []

        for i in tqdm(range(seq_num // args.batch_size + 1)):
            input_ids = torch.tensor(prompt, device=args.device).repeat(args.batch_size, 1)
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                early_stopping=True,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                num_return_sequences=2,#overgenerate to ensure we have enough non-empty generated sequences
                no_repeat_ngram_size=2,
            )

            ppl1= cal_perplexity(output_sequences, model)
            ppls_cur.append(ppl1)

            generated_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)

            for g in generated_sequences:
                labels, seq = g[:prompt_length], g[prompt_length:]
                seq = " ".join(seq.split())
                labels = labels.strip().split("\t")
                if seq:
                    all_data.append([seq]+labels)

        if len(all_data) >seq_num:
            all_data = random.sample(all_data,seq_num)
        return all_data,ppls_cur

    with torch.no_grad():
        prompt_counter = collections.Counter()
        data_path = os.path.join(args.input_training_file, "train.csv")
        with open(data_path, encoding='utf-8') as rf:
            csv_reader = csv.reader(rf)
            title = next(csv_reader)

            #label_column_index = [i for i,name in enumerate(title) if "label" in name]

            for line in csv_reader:
                #prompt = "\t".join([line[idx] for idx in label_column_index]) + "\n\n"
                prompt = "Write an email with subject:"
                prompt_counter[prompt] += 1

        ratio_generation_training = args.total_sequences / sum(prompt_counter.values())
        all_sequences = []
        ppls_cur = []

        for prompt_text in tqdm(prompt_counter):
            prompt = tokenizer(prompt_text)['input_ids']
            num_seq_to_generate = round(prompt_counter[prompt_text] * ratio_generation_training)
            if num_seq_to_generate>0:
                sequences, ppls = generate_text(prompt, num_seq_to_generate, len(prompt_text))
                all_sequences += sequences
                ppls_cur += ppls

    logger.info(f"Current PPL: %.2f±%.2f", np.mean(ppls_cur),np.std(ppls_cur))
    logger.info(f"Total generated sequences: %d", len(all_sequences))
    random.shuffle(all_sequences)

    #prefix = list(filter(None, args.model_name_or_path.split("/"))).pop()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, str(args.length) + "." + str(args.seed) + ".generations.csv")
    with open(output_path, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        for obj in all_sequences:
            if obj[0]: # remove empty sequences
                csv_writer.writerow(obj)

if __name__ == "__main__":
    main()