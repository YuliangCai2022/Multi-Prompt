#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=16GB

mamba init bash
source ~/.bashrc

conda activate climb_adapter

export TOKENIZERS_PARALLELISM=false

python -m run --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks cocoqa,nlvr2,okvqa,snli-ve,gqa\
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		    --do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 16 \
                        --task_attention 0 \
                        --dytox 0 \
                        --ewc 0 \
                        --parallel 0 \
                        --replay 0 \
                        --adapter 0 \
                        --adapter_config_file /project/rostamim_919/caiyulia/Multi-Dytox/src/Vanilla_Houlsby.yaml \
                        --device cuda:0 \
                        --prompt 1 \
                        --freeze_num 6 \
                        #--do_wandb_logging