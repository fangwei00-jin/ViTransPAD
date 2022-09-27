import torch
import argparse
import os
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

rank = args.local_rank
size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

dist.init_process_group("gloo", rank=rank, world_size=size)

x = torch.arange(10)[rank::size].sum()

print("Process rank {}, partial result {}".format(rank, x))

dist.reduce(x, dst=0)
if rank == 0: print("Final result:", x)