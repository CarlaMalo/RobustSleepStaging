#!/usr/bin/env python
if __name__ == '__main__':
    print("Launching training script...", flush=True)
    from train.trainer import train, parse_args
    args = parse_args()
    train(args)
