
# Compositional Behavior Synthesis (CBS) — Zero-Shot Motion Forecasting

**TL;DR:** Predict the future paths of nearby vehicles and pedestrians (3–8s ahead) by learning a small set of **driving primitives** and a **temporal composer** that blends them into full trajectories. The goal is strong **zero-shot** robustness on rare, long-tail scenes.

---

## Results (Waymo Motion v1.0)

**Standard validation (representative parity):**

| Model          | minADE ↓ | minFDE ↓ |
| -------------- | -------: | -------: |
| TNT (baseline) |   0.82 m |   1.45 m |
| **CBS (ours)** |   0.85 m |   1.51 m |

**Curated examples (degradation behavior):**

* **Known / in-distribution:** TNT **1.823 m** vs **CBS 4.231 m** (TNT better on memorized patterns).
* **Novel / long-tail:** TNT **4.156 m** vs **CBS 2.587 m** → **−1.569 m** FDE gap (CBS degrades less).

> **Takeaway:** Parity on common scenes; **much better robustness** on unseen combinations of behaviors.

---

## Repository layout

```
final_submission/
├─ cbs_model_enhanced.py       # Encoders, 4 primitive decoders, temporal composer, confidence head
├─ primitive_detector_v2.py    # Primitive labels + long-tail/challenge-set utilities
├─ waymo_tfexample_loader.py   # Waymo Motion TFExample → vectorized/PKL cache
├─ visualize_training_data.py  # Quick BEV visualizations for data sanity checks
├─ inspect_pkl_data.py         # Inspect cached samples (agents/map/labels)
├─ train_enhanced.py           # Training loop (WTA loss, disentangle, augment)
├─ evaluate.py                 # minADE/minFDE, top-K, per-scene CSVs
├─ run_cbs_system.py           # End-to-end inference + optional visualization
├─ config.py                   # Central hyper-parameters & defaults
└─ outputs/                    # Checkpoints, logs, metrics, figures
```

> If your repo root is different, keep the `final_submission/` prefix in the commands below.

---

## Quickstart

### 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install numpy pandas scipy shapely protobuf tfrecord tensorflow tqdm rich pyyaml opencv-python matplotlib
```

### 2) Data (Waymo Open Dataset — Motion v1.0)

1. Request access and download the **motion** split TFRecords.
2. Set your path and prepare caches:

```bash
export WOD_ROOT=/path/to/waymo_motion_tfrecords
mkdir -p data outputs

# Build cached PKL files (faster training)
python final_submission/waymo_tfexample_loader.py \
  --root $WOD_ROOT --split train --out data/train.pkl

python final_submission/waymo_tfexample_loader.py \
  --root $WOD_ROOT --split val   --out data/val.pkl
```

**(Optional) Visual sanity check**

```bash
python final_submission/visualize_training_data.py \
  --pkl data/train.pkl --num_scenes 16 --out outputs/viz_train
```

### 3) Train CBS

```bash
python final_submission/train_enhanced.py \
  --train_pkl data/train.pkl \
  --val_pkl   data/val.pkl \
  --k 6 --batch_size 64 --epochs 30 --lr 3e-4 \
  --stitch_prob 0.3 --swap_prob 0.3 \
  --disentangle_weight 0.5 \
  --save_dir outputs/cbs_run1 --device cuda
```

> Use `--device cpu` if CUDA isn’t available. Defaults are also set in `config.py`.

### 4) Evaluate (standard validation)

```bash
python final_submission/evaluate.py \
  --val_pkl data/val.pkl \
  --ckpt outputs/cbs_run1/best.pt \
  --k 6 \
  --save_dir outputs/cbs_run1/metrics
```

Artifacts: minADE/minFDE summary, per-scene CSVs, optional PNG renders.

### 5) Zero-Shot / Long-Tail Challenge

Build a list of rare scenarios (e.g., yield→turn, occlusions, odd topology) and evaluate on it.

```bash
# Curate challenge scenes from the val cache
python final_submission/primitive_detector_v2.py \
  --val_pkl data/val.pkl \
  --min_interactions 3 \
  --rare "yield,turn,lane_change" \
  --out data/challenge_list.txt

# Evaluate CBS on that list
python final_submission/evaluate.py \
  --val_pkl data/val.pkl \
  --scene_list data/challenge_list.txt \
  --ckpt outputs/cbs_run1/best.pt \
  --k 6 \
  --save_dir outputs/cbs_run1/metrics_challenge
```

### 6) End-to-End Inference + Viz

```bash
python final_submission/run_cbs_system.py \
  --ckpt outputs/cbs_run1/best.pt \
  --pkl  data/val.pkl \
  --num_scenes 20 \
  --out outputs/cbs_run1/infer \
  --draw_weights   # overlay primitive weights over time
```

---

## Model (1-minute overview)

* **Encoders:**

  * Target agent history: **LSTM + positional encodings** (1s @ 10 Hz)
  * Social context: **multi-head attention** over up to 32 agents
  * Map context: **polyline attention** over 64×20 lane polylines

* **Primitives:** Four expert decoders for **LF/LC/Y/T** + a **primitive classifier**.

* **Temporal composer:** Learns time-varying weights (w_j(t)), (\sum_j w_j(t)=1), to blend primitive trajectories into **K=6** modes; a **confidence head** ranks modes.

* **Losses:** Winner-take-all trajectory loss (over K modes), primitive classification (CE), **disentanglement** (keeps experts specialized).

* **Data-centric augmentation:** **stitch** and **swap-segment** in the local frame to teach smooth A→B behavior transitions.

---

## Configuration & Repro

* Most knobs (K, batch size, lr, augment probs, disentangle weight) are in `config.py` and overridable by CLI flags.
* Seeds and deterministic flags are set in training for reproducibility.
* All runs write checkpoints/metrics under `outputs/<run_name>/`.

---

## License & Team

MIT for code. Use of Waymo Open Dataset must follow Waymo’s terms.
Team 06 — **Deeksha Chauhan**, **Moksh Aggarwal**, **Shivam Mahendru**, **Saurabh Suman**.

---

## Citation

```bibtex
@misc{cbs2025,
  title  = {Compositional Behavior Synthesis for Zero-Shot Motion Forecasting},
  author = {Chauhan, Deeksha and Aggarwal, Moksh and Mahendru, Shivam and Suman, Saurabh},
  year   = {2025},
  note   = {GitHub repository}
}
```
