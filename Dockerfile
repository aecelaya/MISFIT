ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install MISFIT from PyPI. The base image is pinned above (torch 2.9.1) and
# must not be overridden by the build workflow — newer torch has crashed
# multi-GPU DDP startup.
RUN pip install --no-cache-dir misfit-medical

# Create app directory.
RUN mkdir /app
WORKDIR /app
