from llava.train.train import train
import wandb
import os

# wandb.init(project="RL4VLM", name="finetune")

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
