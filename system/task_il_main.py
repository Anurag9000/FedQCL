#!/usr/bin/env python3
"""CLI adapter used by the existing campaign runner."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("-dev","--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("-did","--gpu-id",default="0")
    p.add_argument("--sync-resume",action="store_true")
    p.add_argument("--stop-after-round",type=int)
    a=p.parse_args()
    import torch
    from flcore.sync.task_il_engine import run
    device=torch.device("cpu" if a.device=="cpu" else "cuda:"+a.gpu_id)
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run(json.loads(a.config.read_text()),json.loads(a.schedule.read_text()),
        a.data,a.output,device,a.sync_resume,a.stop_after_round)
if __name__=="__main__":main()
