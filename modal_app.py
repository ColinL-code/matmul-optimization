"""
Usage: modal run modal_app.py --file bench/bench.cu
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("matmul-optimization")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("build-essential")
    .add_local_dir(Path(__file__).parent / "kernels", remote_path="/root/kernels")
    .add_local_dir(Path(__file__).parent / "bench", remote_path="/root/bench")
)


@app.function(image=image, gpu="L4", timeout=10 * 60)
def compile_and_run(path: str) -> str:
    binary = "/tmp/kernel_bin"
    compiled = subprocess.run(
        ["nvcc", "-O3", "-lcublas", f"/root/{path}", "-o", binary],
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        return f"COMPILE ERROR:\n{compiled.stderr}"
    ran = subprocess.run([binary], capture_output=True, text=True)
    return ran.stdout + ran.stderr


@app.local_entrypoint()
def main(file: str) -> None:
    print(compile_and_run.remote(file))
