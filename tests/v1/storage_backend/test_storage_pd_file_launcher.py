# SPDX-License-Identifier: Apache-2.0
"""File-target launcher admission without starting a model or touching a device."""

# Standard
from pathlib import Path
import json
import os
import subprocess

# Third Party
import pytest

LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / "examples/disagg_prefill/1p1d_raw_block/launch_vllm.sh"
)


def launch_file(
    tmp_path: Path,
    target: Path,
    *,
    confirmed: bool = True,
    raw_device: str = "",
    provider: str = "native",
) -> subprocess.CompletedProcess[str]:
    """Run the real launcher with a configuration-printing vLLM stub."""
    executable = tmp_path / "vllm"
    executable.write_text(
        "#!/usr/bin/env python3\nimport os\nprint(os.environ['LMCACHE_EXTRA_CONFIG'])\n"
    )
    executable.chmod(0o700)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LMCACHE_")
    }
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "LMCACHE_RAW_FILE": str(target),
            "LMCACHE_CONFIRM_FILE_OVERWRITE": str(target) if confirmed else "",
            "LMCACHE_STORAGE_PD_SESSION": "test-file-session",
            "LMCACHE_RAW_DEVICE": raw_device,
            "LMCACHE_GPU_BUFFER_PROVIDER": provider,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), "writer", "test-model"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("provider", ["native", "vulkan_rm"])
def test_file_launcher_keeps_strict_registration_and_quotes_paths(
    tmp_path: Path,
    provider: str,
) -> None:
    target = tmp_path / 'cache "with spaces".bin'
    target.write_bytes(bytes(4096))
    result = launch_file(tmp_path, target, provider=provider)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["rust_raw_block.device_path"] == str(target)
    assert config["rust_raw_block.require_dmabuf_registration"] is True
    assert config["rust_raw_block.gpu_buffer_provider"] == provider
    assert "rust_raw_block.allow_unsafe_pd_io_for_testing" not in config
    assert target.read_bytes() == bytes(4096)


@pytest.mark.parametrize(
    "case", ["unconfirmed", "absent", "symlink", "both", "provider"]
)
def test_file_launcher_rejects_wrong_target(tmp_path: Path, case: str) -> None:
    target = tmp_path / "cache.bin"
    if case != "absent":
        target.write_bytes(bytes(4096))
    if case == "symlink":
        link = tmp_path / "link.bin"
        link.symlink_to(target)
        target = link
    result = launch_file(
        tmp_path,
        target,
        confirmed=case != "unconfirmed",
        raw_device="/dev/forbidden" if case == "both" else "",
        provider="invalid" if case == "provider" else "native",
    )
    assert result.returncode == 1
    assert result.stdout == ""
