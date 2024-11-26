#from huggingface_hub import login
#login("")
import os
import glob
import argparse
import torch
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from dataclasses import dataclass
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments

from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100
MAX_LENGTH = 2048 #1024
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class ROSEdataset(Dataset):
    def __init__(self, json_fn, llama_tokenizer=None):
        if not os.path.exists(json_fn):
            raise ValueError(f"{json_fn} does not exist")
        self.inputs = json.load(open(json_fn, "r"))
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):  
        item = self.inputs[index]  
        input = item['input']  # contain promt
        target = item['target']  # cot process answer confidence
        print(f"input: {input}")
        print(f"target: {target}")
        val = self.tokenize(input, target)
        return val
    
    
    def tokenize(self, inp, tar):  
        tokens, prompt_length = self.conditional_generation_task(inp=inp, tar=tar)  
        input_ids = tokens.input_ids[0]     
        labels = tokens.input_ids[0].clone()  # Clone the input_ids for labels  
        # Set the labels for the prompt part to IGNORE_INDEX so they are ignored in loss calculation  
        
        labels[:prompt_length] = IGNORE_INDEX  
        input_id_lens = label_lens = (  
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()  
        )  
        return dict(  
            input_ids=input_ids,  
            input_id_lens=input_id_lens,  
            labels=labels,  
            label_lens=label_lens,  
        )  

    
    def conditional_generation_task(self, inp, tar):  
        
        prompt = inp
        full_text = prompt + tar + self.llama_tokenizer.eos_token  
        tokens = self.llama_tokenizer(  
            full_text,  
            max_length=MAX_LENGTH,  
            return_tensors="pt",  
            truncation=True,  
        )  
        prompt_length = len(self.llama_tokenizer(prompt)['input_ids'])  
        return tokens, prompt_length  


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances)
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_datasets(args, llama_tokenizer, transform_args={}):
    datasets = {
        "train": ROSEdataset(
            str(args.data_path / "train.json"),
            llama_tokenizer=llama_tokenizer,
        ),
        "val": ROSEdataset(
            str(args.data_path / "val.json"),
            llama_tokenizer=llama_tokenizer,
        ),
    }

    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False
    )
    return training_args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def setup_model(args, rank):
    if args.model_name=="llama3":
        model_id = "LLM_Model/Meta-Llama-3-8B" # local path
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map={"": rank})
        llama_tokenizer = pipeline.tokenizer
        model = pipeline.model
    elif args.model_name=="llama2":
        '''llama_options = args.model_name.split("-")
        is_chat = len(llama_options) == 2
        model_size = llama_options[0]

        def llama2_model_string(model_size, chat):
            chat = "chat-" if chat else ""
            return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"'''

        model_string = "model/Llama-2-13b-chat-hf/"
        model = LlamaForCausalLM.from_pretrained(
            model_string,
            load_in_8bit=args.fp8,
            device_map={"": rank},
        )

        llama_tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )
        
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer

def setup_trainer(args):
    training_args = setup_training_args(args)
    model, llama_tokenizer = setup_model(args, training_args.local_rank)

    datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="exp")
    parser.add_argument("--model-name", default="llama3")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=500, type=int)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "ROSE_LLM"
    main(args)
