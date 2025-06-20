#!/usr/bin/env python
# coding=utf-8

from utils.config  import get_config
from solver.solver import Solver

if __name__ == '__main__':
    cfg = get_config('option_GF2.yml')
    solver = Solver(cfg)
    solver.run()
    
