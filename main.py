#!/usr/bin/env python
# coding=utf-8

from utils.config import get_config
from solver.solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='./option_QB_2.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    solver.run()