# RoSe


## environment configuration
The environment configuration can be referred to environments.yaml:

> conda env create -f environments.yaml


# quick start

## fine-tune
> CUDA_VISIBLE_DEVICES=&lt;gpu_ids&gt; accelerate launch --config_file ds_config.yaml llama_finetune.py --run-name &lt;run_name&gt; --data-path &lt;data_path&gt; --eval-freq 200000 --save-freq 50000


## evaluate 
> bash eval.sh

---

Data formats for fine-tuning LLMs include "input" and "target", which can refer to examples.json
