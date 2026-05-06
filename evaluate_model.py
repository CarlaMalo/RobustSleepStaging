#!/usr/bin/env python

if __name__ == '__main__':
    from eval.evaluator import evaluate, parse_args
    args = parse_args()
    evaluate(args)
