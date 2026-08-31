# Rep-LDM Studio — GPU inference container
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# cmake + g++ are needed to build dlib; libgl/libglib for opencv-free PIL/dlib image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY diffae/ diffae/
COPY tools/ tools/
COPY annotations/ annotations/

# Model weights, attribute vectors, outputs and encoder caches are mounted as volumes
ENV REPLDM_CHECKPOINT_DIR=/workspace/checkpoints \
    REPLDM_ATTRIBUTE_DIR=/workspace/attributes \
    REPLDM_OUTPUT_DIR=/workspace/outputs \
    TORCH_HOME=/workspace/cache/torch \
    HF_HOME=/workspace/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
