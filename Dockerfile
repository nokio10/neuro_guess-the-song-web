FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# ШАГ 1: Мелкие зависимости (чтобы не было тайм-аутов)
RUN pip install --no-cache-dir typing_extensions filelock sympy networkx jinja2 fsspec

# ШАГ 2: Устанавливаем PyTorch + Audio + Vision (CPU версии)
# Добавили torchvision==0.23.0+cpu
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==2.8.0+cpu \
    torchaudio==2.8.0+cpu \
    torchvision==0.23.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ШАГ 3: Создаем constraints.txt (добавили туда vision)
# Запрещаем pip'у менять эти версии на что-либо другое
RUN echo "torch==2.8.0+cpu" > constraints.txt && \
    echo "torchaudio==2.8.0+cpu" >> constraints.txt && \
    echo "torchvision==0.23.0+cpu" >> constraints.txt

# ШАГ 4: Устанавливаем остальное
COPY requirements.txt .

# Флаг -c constraints.txt гарантирует, что whisperx или audio-separator
# не смогут скачать "обычный" torch/torchvision и сломать сборку.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints.txt

COPY . .
RUN mkdir -p media temp_downloads
EXPOSE 5001

CMD ["python", "generator_service.py"]
