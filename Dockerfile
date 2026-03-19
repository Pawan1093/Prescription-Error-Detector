FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

CMD ["python", "app_gradio.py"]
```

Also update `requirements.txt` to include gradio:
```
easyocr
opencv-python-headless
torch
torchvision
transformers
accelerate
datasets
evaluate
seqeval
gradio==4.44.0
numpy
Pillow
```

Also update the `README.md` header — change `sdk: gradio` to `sdk: docker`:
```
---
title: Prescription Error Detector
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---