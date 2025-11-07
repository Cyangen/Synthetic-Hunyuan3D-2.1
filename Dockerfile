# Minimal Dockerfile mirroring manual uv-based setup steps
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="hunyuan3d21-backend" \
      maintainer="hunyuan3d21" \
      description="Hunyuan3D-2.1 Model Backend API Server"

# Core environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PYOPENGL_PLATFORM=egl \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace

# Install system packages required by manual script
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    git-lfs \
    unzip \
    cmake \
    ninja-build \
    pkg-config \
    libeigen3-dev \
    libcgal-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxrender1 \
    libxrender-dev \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    mesa-utils-extra \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (exact tooling used in manual steps)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy repository
COPY . /workspace/Hunyuan3D-2.1
WORKDIR /workspace/Hunyuan3D-2.1

# 1. Create venv via uv (Python 3.10)
RUN uv venv --python 3.10 --seed
ENV VIRTUAL_ENV=/workspace/Hunyuan3D-2.1/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# 2. Install CUDA-enabled PyTorch stack using uv
RUN uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 2.5. Install bpy from Blender's PyPI mirror (Python 3.10 compatible)
RUN uv pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/

# 3. Install Python dependencies (same command as setup guide)
RUN uv pip install --index-strategy unsafe-best-match -r requirements.txt

# 4. Install custom_rasterizer with editable install (no build isolation)
RUN cd hy3dpaint/custom_rasterizer && \
    uv pip install --no-build-isolation -e .

# 5. Compile DifferentiableRenderer using pybind11
RUN cd hy3dpaint/DifferentiableRenderer && \
    c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python -m pybind11 --includes) \
    mesh_inpaint_processor.cpp \
    -o mesh_inpaint_processor$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 6. Download Real-ESRGAN checkpoint
RUN mkdir -p hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -P hy3dpaint/ckpt

# 7. Prepare cache directory used for outputs
RUN mkdir -p gradio_cache

# Runtime tuning
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose API and add health check
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Default command: run FastAPI backend used by frontend (v2.1 only public model available)
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8081", "--low_vram_mode"]

