from xmlrpc import client

import pytest
import os
import time
from src import dataloaders, utils, model, train
from src.inference_api import app
import torch
import random
from fastapi.testclient import TestClient

# python.exe -m pytest tests/test_e2e.py

def test_fast_inference():
    conv_model = utils.load_best_model()
    conv_model.eval()
    for i in range(10):
        start = time.time()
        conv_model.generate_number(i)
        assert time.time() - start < 0.1, f"Generación de {i} tomó demasiado tiempo"

def test_generate_endpoint():
    with TestClient(app) as client:
        for i in range(10):
            response = client.post("/generateNumber", json={"number": i})
            assert response.status_code == 200, f"Error al generar número {i}"
        
        response = client.post("/generateNumber", json={"number": -1})
        assert response.status_code != 200

        response = client.post("/generateNumber", json={"number": 10})
        assert response.status_code != 200