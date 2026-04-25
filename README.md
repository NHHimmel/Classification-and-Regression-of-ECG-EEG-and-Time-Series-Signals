# Time Series Classification & Regression with CNN / RNN / LSTM / GRU

A PyTorch study comparing four deep learning architectures on four time series benchmark datasets — two classification tasks and two regression tasks. Experiment metrics are tracked with Weights & Biases.

---

## Datasets

| Dataset | Task | Channels × Length | Classes / Output |
|---|---|---|---|
| **AbnormalHeartbeat** | Binary classification | 1 × 18 530 | 2 (normal / abnormal) |
| **SelfRegulationSCP2** | Binary classification | 7 × 1 152 | 2 |
| **AppliancesEnergy** | Regression | 24 × 144 | 1 (energy consumption) |
| **LiveFuelMoistureContent** | Regression | 7 × 365 | 1 (moisture %) |

Raw data is expected as `.npy` files under a common base path:

```
<BASE_PATH>/
  AbnormalHeartbeat/
    x_train_original.npy  y_train.npy
    x_test_original.npy   y_test.npy
  SelfRegulationSCP2/  ...
  AppliancesEnergy/    ...
  LiveFuelMoistureContent/ ...
```

---

## Models

| Name | Architecture |
|---|---|
| `cnn` | 1-D Conv stack → Global flatten → FC head |
| `rnn` | Vanilla RNN (optionally with CNN downsampler front-end) |
| `lstm` | LSTM (same structure) |
| `gru` | GRU (same structure) |

All four are built through a single factory:

```python
from utils.model_factory import build_model

model = build_model(model_name="lstm", dataset_name="abnormal_heartbeat")
```

Per-dataset architectures (channels, kernel sizes, hidden sizes, dropout …) live in [utils/config.py](utils/config.py).

---

## Project Structure

```
.
├── experiments/
│   ├── AbnormalHeartbeat_01.py      # classification run
│   ├── SelfRegulationSCP2_01.py     # classification run
│   ├── AppliancesEnergy_01.py       # regression run
│   └── LiveFuelMoistureContent_01.py# regression run
├── models/                          # saved .pt checkpoints
├── utils/
│   ├── config.py         # dataset & model hyper-parameters
│   ├── data_loader.py    # load → split → preprocess → DataLoader
│   ├── preprocessor.py   # per-channel z-score, label encoder, target scaler
│   ├── model_factory.py  # CNN1d / RNNModel / RNNWithDS builders
│   └── trainer.py        # ClassificationTrainer & RegressionTrainer
├── main.py               # quick dataset shape inspection
└── pyproject.toml
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and requires Python ≥ 3.13.

```bash
uv sync          # creates .venv and installs all dependencies
```

Dependencies: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `wandb`.

---

## Running Experiments

Edit the `BASE_PATH` variable in the relevant experiment script to point to your data directory, then run:

```bash
# Classification
uv run experiments/AbnormalHeartbeat_01.py
uv run experiments/SelfRegulationSCP2_01.py

# Regression
uv run experiments/AppliancesEnergy_01.py
uv run experiments/LiveFuelMoistureContent_01.py
```

Each script trains CNN, RNN, LSTM, and GRU sequentially and prints a comparison table at the end:

```
============================================================
Test Results — abnormal_heartbeat
============================================================
Model    |  Test Loss |   Test Acc
------------------------------------
CNN      |     0.2341 |     91.23%
RNN      |     0.3102 |     88.46%
...
Best model: CNN (acc: 91.23%, loss: 0.2341)
```

Runs are logged to W&B automatically. Set `use_wandb=False` in the trainer call to disable.

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (Adam for regression) |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 epochs |
| LR scheduler | `ReduceLROnPlateau` (classification) |

- **Classification metric:** accuracy; loss = `CrossEntropyLoss` (weighted for class imbalance where applicable).
- **Regression metric:** NRMSE (RMSE normalised by target std); loss = `MSELoss`.
- Best checkpoint (lowest validation loss) is restored automatically before evaluation.

---

## Preprocessing

Handled inside `load_dataset()`:

1. Stratified train/val split (classification) or random split (regression).
2. Per-channel z-score normalisation of features (datasets that require it).
3. Target z-score normalisation (regression datasets).
4. Label encoding for classification targets.
