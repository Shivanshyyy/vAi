# syntax=docker/dockerfile:1
#==============================================================================
# vAi environment image: run the C++ demos and pyvai.
#
#   docker build -f docker/Dockerfile -t vai .      # from the repo root
#
# Inside: `cmake --build build` rebuilds the C++ demos, and
# `uv pip install -e .` refreshes the bindings.
#==============================================================================

# Pin the digest for a tagged release: ubuntu:24.04@sha256:...
ARG BASE_IMAGE=ubuntu:24.04

# LunarG publishes one apt repo per SDK release — this ARG is the SDK pin.
ARG VULKAN_SDK_VERSION=1.4.313
ARG UBUNTU_CODENAME=noble
ARG UV_VERSION=0.12.7


# Named so the pin above drives it; uv ships as a static binary.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


################################################################################
# base — Vulkan runtime: loader + in-image ICD
################################################################################
FROM ${BASE_IMAGE} AS base
ARG VULKAN_SDK_VERSION
ARG UBUNTU_CODENAME
ARG DEBIAN_FRONTEND=noninteractive

# Mesa is the AMD/Intel ICD (NVIDIA's is injected at run time). The x11/egl
# libs are what NVIDIA's ICD needs headless — missing, it silently falls
# back to llvmpipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc \
        -o /etc/apt/trusted.gpg.d/lunarg.asc \
    && curl -fsSL "https://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION}/lunarg-vulkan-${VULKAN_SDK_VERSION}-${UBUNTU_CODENAME}.list" \
        -o /etc/apt/sources.list.d/lunarg-vulkan.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 vulkan-tools mesa-vulkan-drivers \
        libx11-6 libxext6 libegl1 \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# `graphics` is required, or the toolkit skips the Vulkan ICD.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics


################################################################################
# dev — C++ demos + pyvai, built from the source in the image
################################################################################
FROM base AS dev
ARG DEBIAN_FRONTEND=noninteractive

# Toolchain + full SDK (glslc); the LunarG repo list is inherited from base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        build-essential g++ cmake ninja-build \
        python3 python3-dev \
    && apt-get install -y vulkan-sdk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

# The venv the bindings install into (PEP 668 blocks the system one). Built on
# the distro python3, not a uv-managed one: nanobind compiles against the
# headers python3-dev provides.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy
RUN uv venv --python python3 /opt/venv

# /opt/vai is baked into VAI_DATA_DIR / SHADER_OUTPUT_DIR at configure time:
# shaders load from build/, checkpoints mount at data/.
WORKDIR /opt/vai
COPY . .

# The cache mount keeps uv's downloads across builds, so a source change does
# not re-fetch the build backend.
RUN --mount=type=cache,target=/root/.cache/uv \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DVAI_BUILD_EXAMPLES=ON \
    && cmake --build build \
    && uv pip install -e .

ENV PATH=/opt/vai/build/bin:$PATH

CMD ["bash"]
