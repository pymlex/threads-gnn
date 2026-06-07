import torch


def require_torch_scatter() -> None:
    """Abort training when torch-scatter is missing or broken."""
    import torch_scatter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor([1.0, 2.0], device=device)
    index = torch.tensor([0, 0], device=device)
    output = torch_scatter.scatter_sum(values, index, dim=0, dim_size=1)
    if not torch.isfinite(output).all().item():
        raise RuntimeError(
            "torch-scatter is installed but returns non-finite values. "
            "Re-run: bash scripts/install.sh"
        )
