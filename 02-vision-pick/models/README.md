# Trained models

Both files here were trained on this cell, on the 233 labelled frames in
[`../dataset/`](../dataset/), using [`../train_cubes.py`](../train_cubes.py).
One class: `cube`.

| File | Base | Trained | mAP50 | Used by |
|---|---|---|---|---|
| [`../cube_model.pt`](../cube_model.pt) | YOLO26n | 2026-08-10, 80 epochs | **0.979** | the application — `vision_pick3.py` loads this path |
| `cube_model_v8n.pt` | YOLOv8n | 2026-08-02 | see `../training/runs-cubes-v8n/results.csv` | superseded, kept for comparison |

The current model lives one folder up, at `02-vision-pick/cube_model.pt`,
because that is where the application looks for it. Do not move it.

Full training curves, confusion matrices and per-epoch metrics for both
runs are in [`../training/`](../training/).

## Retraining

```
cd 02-vision-pick
pip install -r requirements.txt
python train_cubes.py
```

`train_cubes.py` reads `dataset/data.yaml`, fine-tunes from the existing
`cube_model.pt` rather than from scratch, and writes a new run under
`runs/`. Add more frames first with `launchers\1 - Capture Dataset.bat`.

---

Trained by **Musab Bagazi** and **Yazan Bal'fakeeh**.
