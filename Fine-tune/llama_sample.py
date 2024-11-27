import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import transformers

from transformers import (
    LlamaForCausalLM, LlamaTokenizer
)
from peft import PeftModel

from llama_finetune import (
    MAX_LENGTH
)

from tqdm import tqdm
import jsonlines

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def prepare_model_and_tokenizer(args):
    model = None
    if args.model_name=="llama3":
        model_id = "LLM_Model/Meta-Llama-3-8B" 
        if args.no_ft:
            model_id += "-Instruct"
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, 
                                         # device_map={"": rank}
                                         device_map="auto"
                                        )
        tokenizer = pipeline.tokenizer
        model = pipeline.model
    
    elif args.model_name=="llama2":
        llama_options = args.model_name.split("-")
        is_chat = len(llama_options) == 2
        model_size = llama_options[0]

        def llama2_model_string(model_size, chat):
            chat = "chat-" if chat else ""
            return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

        model_string = llama2_model_string(model_size, is_chat)
    
        model = LlamaForCausalLM.from_pretrained(
            model_string,
            load_in_8bit=True,
            device_map="auto",
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )
    
    if not args.no_ft:
        print("Loading LoRA Weights~~~~~~")
        model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")
    else:
        print("Use origin LLaMA~~~~~~")
    print(f"Model: {model_id}")
    
    return model, tokenizer

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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def eval_sample(args, model, tokenizer):
   
    if not os.path.exists(args.test_json_fn):
        raise ValueError(f"File {args.test_json_fn} does not exist")
    if '.jsonl' in args.test_json_fn:
        with jsonlines.open(args.test_json_fn, 'r') as reader:
            data = [line for line in reader]
    else:
        with open(args.test_json_fn, 'r') as file:
            data = json.load(file)
    
    output_path = args.out_path + 'l' 
    
    exist_ids = []
    if os.path.exists(output_path):
        with jsonlines.open(output_path, 'r') as reader:
            for line in reader:
                exist_ids.append(line['id'])
        print(f'Already have {len(exist_ids)} data~')
    
    
    # outputs = []
    # answer_format 
    with jsonlines.open(output_path, "a") as writer:
        for item in tqdm(data, desc=f"File: {args.out_path.replace('.json', '')}"):
            if item['id'] in exist_ids: # jump some files already exists
                continue
            input_text = item['input']
            
            # if args.no_ft:
            #     input_text = input_text.replace(", analysis: {analysis}", "")
            #     input_text = input_text.replace("\nstep1:", "\n请参考以下步骤回答问题:\nstep1:")
            #     input_text = input_text.replace("回答格式为step: {step}, answer: {answer}, confidence: {confidence}", answer_format)
            
            prompt = input_text 

         
            batch = tokenizer([prompt], return_tensors="pt")
            batch = {k: v.cuda() for k, v in batch.items()}


            generate_ids = model.generate(
                **batch,
                do_sample=args.do_sample, # args.do_sample,
                max_new_tokens=args.max_tokens,  # 500
                temperature=args.temperature, # args.temperature,
                top_p=args.top_p,
            )

            gen_str = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

            material_str = gen_str.replace(prompt, "")

            output = {
                "id": item['id'],
                "input": input_text,
                "prediction": material_str,
                "model_name": args.model_name,
                "is_ft": not args.no_ft,
            }
            # print(f"Input: {input_text}")
            writer.write(output)


    with open(output_path, 'r', encoding='utf-8') as infile, open(args.out_path, 'w', encoding='utf-8') as outfile:
        json_objects = [json.loads(line) for line in infile]
        # write json
        json.dump(json_objects, outfile, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="output/")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=500)

    parser.add_argument("--test_json_fn", type=str, default="data/") 
    parser.add_argument("--no_ft", action="store_true", help="if true, use llama w/o ft")
    args = parser.parse_args()
    
    if args.temperature == 0:
        args.do_sample = False
    else:
        args.do_sample = True

    if ".json" in args.out_path:
        out_path = args.out_path
    else:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(args.out_path, f"samples_{i}.json") 
        args.out_path = out_path
        
    if args.no_ft:
        out_dir = 'output'
    else:
        out_dir = 'output_ft'
        

    model, tokenizer = prepare_model_and_tokenizer(args)
    
    paths = ["eval-ft-id", "eval-ft-ood"]
  
    for path in paths: 
        args.test_json_fn = f"data/{path}.json"
        args.out_path = f"{out_dir}/{path}.json"
        eval_sample(args, model, tokenizer)
        
