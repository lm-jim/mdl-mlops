import pytest
import os
import time
from src import dataloaders, utils, model, train
import torch
import random

# python.exe -m pytest tests/test_e2e.py

def test_fast_inference():
    conv_model = model.load_best_model()
    conv_model.eval()
    for i in range(10):
        start = time.time()
        conv_model.generate_number(i)
        assert time.time() - start < 0.1, f"Generación de {i} tomó demasiado tiempo"
