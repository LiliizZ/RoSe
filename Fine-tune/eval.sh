gpu_ids=4
model_name=llama3
model_path=./checkpoint-40000 ## fine-tuned model
temperature=0.9
max_tokens=500

# CUDA_VISIBLE_DEVICES=gpu_ids accelerate launch --config_file ds_config.yaml llama_finetune.py \
CUDA_VISIBLE_DEVICES=$gpu_ids python llama_sample.py \
    --model_path $model_path \
    --model_name $model_name \
    --temperature $temperature \
    --max_tokens $max_tokens \
    --no_ft
   
    
