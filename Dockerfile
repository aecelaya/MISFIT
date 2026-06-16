ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install MISFIT.
RUN pip install --no-cache-dir misfit-medical

# Create app directory.
RUN mkdir /app
WORKDIR /app
