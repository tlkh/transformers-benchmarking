import time
script_start = time.time()

import os
import torch
import pytorch_lightning as pl
from datasets import load_dataset
import transformers
import models
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=128)
parser.add_argument("--dataset", type=str, default="imdb")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--profile_mode", action="store_true")
parser.add_argument("--pytorch_adamw", action="store_true", help="Use PyTorch instead of Apex optimizer")
parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
parser.add_argument("--asp", action="store_true", help="Enable 2:4 Automatic Sparsity")
parser.add_argument("--exp_name", type=str, default="benchmark")
args = parser.parse_args()

if args.pytorch_adamw:
    print("Not using Apex Fused AdamW Optimizer. Slower training is expected.")
    fused_opt = False
else:
    fused_opt = True

def load_data(args, tokenizer, split="train"):
    dataset = load_dataset(args.dataset, split=split)
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_seq_len, padding=True), batched=True)
    dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "label"])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=args.num_workers)
    
    return dataloader

def run_training(args, profiler, trainer_precision, do_eval=True):
    transformer_model = models.TransformerModel(model_name=args.model_name, num_labels=2, fused_opt=fused_opt, sparse=args.asp)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataloader = load_data(args, tokenizer, split="train")
    test_dataloader = load_data(args, tokenizer, split="test")
    
    train_start = time.time()
    cold_start_time = train_start-script_start
    print("Cold start time:", int(cold_start_time), "(can use this to set delay for profiling)")
    
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.num_gpus, accelerator="ddp",
                         amp_backend="native", precision=trainer_precision, progress_bar_refresh_rate=0,
                         benchmark=True, profiler=profiler, logger=False, checkpoint_callback=False)
    
    print("Start training (progress bar is disabled)")
    start_time = time.time()
    train_results = trainer.fit(transformer_model, train_dataloader)
    end_time = time.time()
    train_duration = end_time-start_time
    print("Finished training in", int(train_duration), "seconds")
    
    results = {
        "cold_start_time": cold_start_time,
        "train_duration": train_duration,
        "train_results": train_results,
    }
    
    print(results)
    
    if do_eval:
        print("Running evaluation:")
        eval_results = trainer.test(transformer_model, test_dataloader, verbose=False)
        results["eval_results"] = eval_results
        
    return results
        
def main():
    OUTPUT_DIR = "./"+args.exp_name+"/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiment_data = vars(args)
    
    experiment_data["vram_start"] = utils.get_gpu_memory_info()
    
    utils.setup_env(profiling=args.profile_mode, output_file=OUTPUT_DIR+"env_info.txt")
    utils.get_gpu_hw_info(output_file=OUTPUT_DIR+"gpu_hw_info.txt")
    profiler = pl.profiler.SimpleProfiler(output_filename=OUTPUT_DIR+"pl_profile.txt")
    if args.amp:
        trainer_precision = 16
    else:
        trainer_precision = 32
    
    results = run_training(args, profiler, trainer_precision, do_eval=True)
        
    experiment_data["vram_end"] = utils.get_gpu_memory_info()
    experiment_data["train_duration"] = results["train_duration"]
    experiment_data["cold_start_time"] = results["cold_start_time"]
    experiment_data["eval_results"] = results["eval_results"]
    
    utils.dump_json(experiment_data, output_file=OUTPUT_DIR+"exp_info.json")
    
if __name__ == "__main__":
    main()
    