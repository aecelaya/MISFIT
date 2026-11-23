# MISFIT builds on the MIST image, which already has MIST, a C compiler for
# torch.compile, cuDNN 9.15+, a default username for numeric UIDs, and a
# writable Triton cache. Keep MIST_IMAGE on a release that MISFIT supports.
# The PyTorch base (torch 2.9.1) is pinned in MIST's Dockerfile: newer torch
# has crashed multi-GPU DDP startup.
ARG MIST_IMAGE=mistmedical/mist:3.0.0
FROM ${MIST_IMAGE}

# Install MISFIT. The publish workflow passes the release's version, so the
# image gets exactly that release. Left empty, a local build installs the
# latest one. Installing MISFIT keeps the image's MIST, which satisfies its
# mist-medical requirement.
#
# The workflow starts right after the PyPI upload, and PyPI's cached index can
# take several minutes to list the new version, so retry every 30 s for up to
# 15 minutes.
#
# Resolving torch's dependencies also puts back the cuDNN that PyTorch 2.9 pins
# (9.10), which runs BF16 3D convolutions about 3x slower; the 0.3.0a0 image
# shipped with it. So upgrade cuDNN again afterwards, as MIST's image does. pip
# warns that torch pins 9.10; PyTorch works with the newer cuDNN.
ARG MISFIT_VERSION=""
RUN if [ -n "$MISFIT_VERSION" ]; then spec="misfit-medical==${MISFIT_VERSION}"; \
    else spec="misfit-medical"; fi; \
    for attempt in $(seq 1 30); do \
        pip install --no-cache-dir "$spec" \
            && pip install --no-cache-dir -U "nvidia-cudnn-cu12>=9.15" \
            && exit 0; \
        echo "pip install $spec failed (attempt $attempt of 30); retrying in 30 s"; \
        sleep 30; \
    done; \
    exit 1

# Fail the build if pip installed a different MISFIT than asked for, replaced
# the image's MIST 3 or its newer cuDNN, or the training CLI can't import.
RUN python -c "import sys, torch; from importlib.metadata import version; v = version('misfit-medical'); m = version('mist-medical'); c = torch.backends.cudnn.version(); want = sys.argv[1]; print('misfit-medical', v, '| mist-medical', m, '| cuDNN', c); assert not want or v == want, (v, want); assert int(m.split('.')[0]) >= 3, m; assert c and c >= 91500, c; import misfit.cli.train_entrypoint" "$MISFIT_VERSION"

WORKDIR /app
