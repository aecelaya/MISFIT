# MISFIT builds on the MIST image, which already has MIST, a C compiler for
# torch.compile, cuDNN 9.15+, a default username for numeric UIDs, and a
# writable Triton cache. Keep MIST_IMAGE on a release that MISFIT supports.
# The PyTorch base (torch 2.9.1) is pinned in MIST's Dockerfile: newer torch
# has crashed multi-GPU DDP startup.
ARG MIST_IMAGE=mistmedical/mist:3.0.0rc0
FROM ${MIST_IMAGE}

# Install MISFIT. The publish workflow passes the release's version, so the
# image gets exactly that release. Left empty, a local build installs the
# latest one. Installing MISFIT keeps the image's MIST, which satisfies its
# mist-medical requirement.
ARG MISFIT_VERSION=""
RUN if [ -n "$MISFIT_VERSION" ]; then \
        pip install --no-cache-dir "misfit-medical==${MISFIT_VERSION}"; \
    else \
        pip install --no-cache-dir misfit-medical; \
    fi

# Fail the build if pip installed a different MISFIT than asked for, replaced
# the image's MIST 3, or the training CLI can't import.
RUN python -c "import sys; from importlib.metadata import version; v = version('misfit-medical'); m = version('mist-medical'); want = sys.argv[1]; print('misfit-medical', v, '| mist-medical', m); assert not want or v == want, (v, want); assert int(m.split('.')[0]) >= 3, m; import misfit.cli.train_entrypoint" "$MISFIT_VERSION"

WORKDIR /app
