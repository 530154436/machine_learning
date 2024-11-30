#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(BASE_DIR.joinpath('data'))

print(DATA_DIR)
