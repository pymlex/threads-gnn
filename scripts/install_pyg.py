import subprocess
import sys

import torch


def cuda_tag() -> str:
    """Return PyG CUDA wheel tag for the active PyTorch build."""
    cuda = torch.version.cuda
    if cuda is None:
        return "cpu"
    return "cu" + cuda.replace(".", "")


def torch_wheel_tags() -> list[str]:
    """Build candidate PyG wheel index tags for the active PyTorch version."""
    raw = torch.__version__.split("+")[0]
    parts = raw.split(".")
    major = parts[0]
    minor = parts[1]
    patch = parts[2] if len(parts) > 2 else "0"
    patch = patch.split("a")[0].split("b")[0].split("rc")[0]
    exact = f"{major}.{minor}.{patch}"
    minor_zero = f"{major}.{minor}.0"
    cuda = cuda_tag()
    tags = []
    for version in [exact, minor_zero]:
        tag = f"torch-{version}+{cuda}"
        if tag not in tags:
            tags.append(tag)
    return tags


def pip_run(args: list[str]) -> None:
    """Run pip with the current Python interpreter."""
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def scatter_available() -> bool:
    """Return whether torch-scatter loads and runs on the active device."""
    import torch_scatter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor([1.0, 2.0], device=device)
    index = torch.tensor([0, 0], device=device)
    output = torch_scatter.scatter_sum(values, index, dim=0, dim_size=1)
    return bool(torch.isfinite(output).all().item())


def main() -> None:
    """Install PyG CUDA extensions matched to the Colab PyTorch build."""
    pip_run([
        "uninstall",
        "-y",
        "torch-scatter",
        "torch-sparse",
        "pyg-lib",
        "torch-cluster",
        "torch-spline-conv",
    ])

    for tag in torch_wheel_tags():
        index = f"https://data.pyg.org/whl/{tag}.html"
        pip_run([
            "install",
            "--no-cache-dir",
            "pyg-lib",
            "torch-scatter",
            "torch-sparse",
            "-f",
            index,
        ])
        if scatter_available():
            print(f"PyG extensions installed from {index}")
            return
        pip_run(["uninstall", "-y", "torch-scatter", "torch-sparse", "pyg-lib"])

    raise SystemExit(
        "torch-scatter installation failed. "
        f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}. "
        "Run: python -c \"import torch; print(torch.__version__, torch.version.cuda)\""
    )


if __name__ == "__main__":
    main()
