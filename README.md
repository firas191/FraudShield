# FraudShield

Federated Learning platform with Differential Privacy for multi-institutional fraud detection. Three simulated banks collaboratively train a fraud detector — raw transaction data never leaves each bank; only DP-noised model updates are shared.

**Status: Week 1 of 8** — environment, model architecture, dataset acquisition, unit tests.

## Quick start (WSL2, Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m data.prepare_dataset download
python -m data.prepare_dataset explore
pytest tests/ -v
```

## Layout

```
model/      # FraudDetectorMLP + safetensors weight utilities
data/       # dataset download/exploration (partitioner: Week 2)
client/     # DP-SGD training engine (Week 2+)
server/     # FastAPI coordination server (Week 3+)
attacks/    # DLG gradient inversion demo (Week 5)
dashboard/  # React + TypeScript frontend (Week 6)
tests/      # pytest suite
```

Full architecture, math, and the 8-week plan: see the project specification document.
