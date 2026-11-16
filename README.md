# Meal Delivery Routing: Attention Model vs. Classical Baselines

Course project (team of 3). This repo contains my part: a learned pickup-and-delivery router
(Kool et al., 2019, adapted) and the shared benchmark pipeline, evaluated on the Grubhub
Meal Delivery Routing Problem instances. The ALNS and OR-Tools rolling-horizon methods
are teammates' work: [link or note].

## Results (100 real held-out chunks per size, 7 unseen seeds)

Cost per order, lower is better.

| Orders per courier | Nearest-neighbor | Insertion | Attention (greedy) |
|---|---|---|---|
| 5  | 133.6 | 101.6 | 101.6 |
| 10 | 225.9 | 191.8 | 192.1 |
| 20 | 413.6 | 371.7 | 375.4 |

The attention model is within 1% of cheapest insertion and 9-24% better than nearest-neighbor,
3.4% above the exact optimum at 5 orders, and about 4x lower latency at 20 orders
(43 vs. 186 ms). Single training run; spread across seeds not yet reported. Work in progress.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/grubhub/mdrplib.git
```

## Run

```bash
python baselines.py mdrplib/public_instances/0o100t100s1p100 greedy solutions/0o100t100s1p100/greedy
python mdrp_chunks.py          # selects 30 instances, builds test chunks and the synthetic generator
python chunk_env.py            # nearest-neighbor, insertion, exact optimum on the chunks
python train_attention.py --steps 3000
python eval_attention.py
```

## Files

- `mdrp_data.py`: instance loader
- `baselines.py`: online nearest-neighbor and greedy insertion, plus the official solution writer
- `mdrp_chunks.py`: instance split (3 calibration / 7 test seeds), test chunks, synthetic generator
- `chunk_env.py`: single-courier cost function and chunk baselines
- `attention_model.py`, `train_attention.py`, `eval_attention.py`: the model

Data: Grubhub MDRP instance library (github.com/grubhub/mdrplib), not included here.