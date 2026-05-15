---
title: MLOps MDL Project
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)

![Tests](https://github.com/lm-jim/mdl-mlops/actions/workflows/tests.yml/badge.svg)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Active-gold.svg)](https://wandb.ai/lm-jim-universidad-polit-cnica-de-madrid/mdl-mlops/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/lm-jim/mdl-mlops)

---

# Despliegue MLOps completo de IA generativa de dígitos MNIST

Proyecto final para la asignatura MLOps en el Máster en Deep Learning por la Universidad Politécnica de Madrid

- Autor: **Luis Miguel Jiménez Aliaga**

## Contenido del proyecto

- Código de modelo generativo de dígitos del MNIST, estructurado al estándar de proyectos ML/DL
- Sistema de configuración y logging
- Versionado y configuraciones de modelos
- Pipeline de entrenamiento
- Código pytest, integrado automáticamente con GitHub Actions, conteniendo:
    - Tests unitarios
    - Test de integración
    - Tests End 2 End
- Integración con Weights & Biases
- Gestión de dependencias por requirements.txt
- Scripts para generar imagen Docker o arrancar sesión local uvicorn
- Despliegue web accesible desde HuggingFace Spaces
- Notebook Jupyter con generaciones de ejemplo

## Instrucciones de ejecución

### Despliegue local con uvicorn

El servidor se puede lanzar ejecutando el script `scripts/launch-uvicorn.bat` (sólo Windows) o introduciendo el siguiente comando en la raíz del proyecto:

> `python -m uvicorn src.inference_api:app --host 127.0.0.1 --port 8000 --reload`

### Despliegue local por contenedor Docker

El servidor se puede construir como imágen Docker utilizando el script `scripts/build-docker.bat` (sólo Windows) o introduciendo el siguiente comando en la raíz del proyecto:

> `docker build -t number-generator-mnist .`

Una vez construido, se puede acceder desde Docker Desktop o introducir el siguiente comando:

> `docker run -p 8000:8000 number-generator-mnist`

Si se accede desde Docker Desktop, es importante que enlacemos al puerto 8000 para tener acceso al servidor.

### Acceso online por HuggingFace Spaces

También se puede acceder al despliegue online desde el siguiente enlace:

> https://huggingface.co/spaces/lm-jim/mdl-mlops

## Enlaces de interés

| Recurso                   | Enlace                                                               |
| :------------------------ | :------------------------------------------------------------------- |
| Repositorio GitHub        | https://github.com/lm-jim/mdl-mlops                                  |
| Proyecto W&B              | https://wandb.ai/lm-jim-universidad-polit-cnica-de-madrid/mdl-mlops/ |
| Despliegue en HuggingFace | https://huggingface.co/spaces/lm-jim/mdl-mlops                       |
