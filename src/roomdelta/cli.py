"""roomdelta doctor | compare | view"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Compare two videos of a room and report what moved, appeared or disappeared.")


@app.command()
def doctor():
    """Check GPU, packages and checkpoint access."""
    import importlib

    ok = True
    try:
        import torch

        cuda = torch.cuda.is_available()
        typer.echo(f"torch {torch.__version__}  CUDA {'yes: ' + torch.cuda.get_device_name(0) if cuda else 'NO'}"
                   + (f"  ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)" if cuda else ""))
        ok &= cuda
    except Exception as e:  # noqa: BLE001
        typer.echo(f"torch: {e}"); ok = False
    for mod in ("depth_anything_3", "lightglue", "transformers", "open3d", "viser", "cv2", "sklearn"):
        try:
            m = importlib.import_module(mod)
            typer.echo(f"  {mod:18s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  {mod:18s} MISSING ({e.__class__.__name__})"); ok = False
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for repo in ("depth-anything/DA3-BASE", "depth-anything/DA3METRIC-LARGE", "IDEA-Research/grounding-dino-base", "facebook/sam2.1-hiera-large", "facebook/dinov2-base"):
            try:
                api.hf_hub_download(repo, "config.json")
                typer.echo(f"  checkpoint {repo:38s} accessible")
            except Exception as e:  # noqa: BLE001
                gated = "gated" in str(e).lower() or "403" in str(e) or "401" in str(e)
                typer.echo(f"  checkpoint {repo:38s} {'GATED (request access; optional)' if gated else 'unavailable: ' + e.__class__.__name__}")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"huggingface_hub: {e}")
    typer.echo("ready" if ok else "NOT READY")
    raise typer.Exit(0 if ok else 1)


@app.command()
def compare(before: Path = typer.Option(..., exists=True), after: Path = typer.Option(..., exists=True),
            output: Path = typer.Option(...), backend: str = "da3-base", keyframes: int = 32, resolution: int = 504,
            meters_per_unit: Optional[float] = typer.Option(None, help="calibrated scale (overrides the approximate prior)"),
            metric_model: str = typer.Option("depth-anything/DA3METRIC-LARGE", help="learned metric depth for scale; 'none' to skip"),
            camera_height: Optional[float] = typer.Option(1.40, help="handheld camera height prior in m (fallback + cross-check); 0 to disable"),
            robot_radius: float = 0.22, geometry_mode: str = typer.Option("shared_pose", help="shared_pose | independent"),
            track: bool = typer.Option(True, help="SAM 2.1 video tracking of objects"), force: bool = False):
    """Run the full pipeline and write events.json, report.md and the viewer cache."""
    from .pipeline import Config, compare as run

    cfg = Config(geometry_backend=backend, max_keyframes=keyframes, long_side=resolution, meters_per_unit=meters_per_unit,
                 camera_height_prior_m=camera_height or None, robot_radius_m=robot_radius,
                 metric_model=None if metric_model.lower() == "none" else metric_model,
                 geometry_mode=geometry_mode, track_objects=track)
    run(str(before), str(after), output, cfg, force=force)
    typer.echo((output / "report.md").read_text())


@app.command()
def view(output: Path = typer.Argument(..., exists=True), port: int = 8080, host: str = "0.0.0.0"):
    """Open the interactive evidence viewer for a finished comparison (no models needed)."""
    from .viewer import serve

    serve(output, host=host, port=port)


if __name__ == "__main__":
    app()
