ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install MISFIT from source. MISFIT is not published to PyPI, so the package
# is built from the repository copied into the image (the build context must be
# the repository root: `docker build -t misfit .`).
WORKDIR /opt/misfit
COPY . /opt/misfit
RUN pip install --no-cache-dir .

# Create app directory.
RUN mkdir /app
WORKDIR /app
