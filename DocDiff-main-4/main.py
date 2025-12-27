# main.py （torchrun 版本）
import os
import torch
import torch.distributed as dist
from src.config import load_config
from src.train import Trainer

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml')
    args = parser.parse_args()

    # ✅ 在每个子进程中独立加载 config（避免 pickle）
    config = load_config(args.config)

    # 从环境变量获取分布式参数
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    config.RANK = local_rank
    config.WORLD_SIZE = world_size

    setup()
    torch.cuda.set_device(local_rank)

    trainer = Trainer(config)
    if config.MODE == 1:
        trainer.train()
    else:
        trainer.test()

    cleanup()

if __name__ == "__main__":
    main()