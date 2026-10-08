# SPDX-License-Identifier: Apache-2.0
"""Explicit-only build of the version-coupled NVIDIA staging provider."""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import os
import subprocess
import sys

# Third Party
from torch.utils import cpp_extension
import torch

if TYPE_CHECKING:
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

NVIDIA_SOURCE = "61dcc93722ecb418bb5f2e00923f05b4b8051dd1"


class VulkanRMStorageBackend(StorageBackendProfile):
    """Build the optional helper; never enable it through auto-detection."""

    name = "vulkan_rm"
    env_var = "BUILD_WITH_VULKAN_RM"

    def detect(self) -> bool:
        """Return false: RM interop requires an explicit build opt-in."""
        return False

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build against clean NVIDIA 615.71.09 headers and CUDA/Vulkan SDKs.

        Raises:
            RuntimeError: The platform, Torch build, or pinned headers do not
                match the experimental profile.
            subprocess.CalledProcessError: Header provenance cannot be checked.
        """
        if sys.platform != "linux" or torch.version.cuda is None:
            raise RuntimeError("Vulkan/RM staging requires Linux and CUDA Torch")
        header_root = os.environ.get("LMCACHE_NVIDIA_RM_HEADERS")
        if not header_root:
            raise RuntimeError("Set LMCACHE_NVIDIA_RM_HEADERS to NVIDIA 615.71.09")
        root = Path(header_root).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
        if revision != NVIDIA_SOURCE or changes:
            raise RuntimeError("RM headers must be a clean NVIDIA 615.71.09 checkout")
        return [
            cpp_extension.CUDAExtension(
                "lmcache._vulkan_rm",
                sources=["csrc/storage_backends/vulkan_rm/staging.cpp"],
                include_dirs=[
                    str(root / "src/common/sdk/nvidia/inc"),
                    str(root / "src/common/inc"),
                    str(root / "kernel-open/common/inc"),
                    str(root / "kernel-open/nvidia"),
                    str(root / "src/nvidia/arch/nvalloc/unix/include"),
                ],
                libraries=["vulkan", "cuda"],
                extra_compile_args={"cxx": extra_cxx_flags + ["-std=c++20"]},
            )
        ]
