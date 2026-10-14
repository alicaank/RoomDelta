# Third-party components

Code and checkpoints are licensed separately. RoomDelta downloads checkpoints at run time; none are redistributed here.

| Component | Used for | Code license | Checkpoint | Checkpoint license |
|---|---|---|---|---|
| Depth Anything 3 (ByteDance-Seed/Depth-Anything-3) | per-visit depth, intrinsics, extrinsics, confidence | Apache-2.0 | `depth-anything/DA3-BASE` (default) | Apache-2.0 |
| | metric scale | | `depth-anything/DA3METRIC-LARGE` | Apache-2.0 |
| | optional larger backends | | `DA3-LARGE(-1.1)`, `DA3-GIANT(-1.1)`, `DA3NESTED-*` | **CC BY-NC 4.0** (non-commercial) |
| LightGlue (cvg/LightGlue) | feature matching | Apache-2.0 | `aliked_lightglue` | Apache-2.0 |
| ALIKED | local features | BSD-3-Clause | `aliked-n16` | BSD-3-Clause |
| Grounding DINO (via Hugging Face transformers) | text-prompted boxes | Apache-2.0 | `IDEA-Research/grounding-dino-base` | Apache-2.0 |
| SAM 2.1 (via transformers) | box / point prompted masks, video tracking | Apache-2.0 | `facebook/sam2.1-hiera-large` | Apache-2.0 |
| DINOv2 (via transformers) | object-level appearance identity | Apache-2.0 | `facebook/dinov2-base` | Apache-2.0 |
| Open3D | registration, plane fitting | MIT | – | – |
| Viser | interactive viewer | Apache-2.0 | – | – |
| PyTorch, NumPy, SciPy, scikit-learn, OpenCV | numerics | BSD / Apache-2.0 | – | – |

Related work (not used as code): SceneDiff (two-video object change detection with asymmetric depth tests), VGGT-CD (cross-session reconstruction alignment).
