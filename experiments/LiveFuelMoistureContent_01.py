import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATASET_CONFIGS, MODEL_CONFIGS
from utils.data_loader import load_dataset
from utils.model_factory import build_model
from utils.trainer import RegressionTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb

BASE_PATH = "C:/Users/AlphaNHH/Downloads/Time Series Data/"
CHK_DIR = Path("./models")
CHK_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["cnn", 'rnn', 'lstm', 'gru']

DATASET_NAME = "live_fuel_moisture"
CONFIG = DATASET_CONFIGS[DATASET_NAME]

LR            = 1e-3
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 32
N_EPOCHS      = 100
EARLY_STOP    = 15

results = []


# Loading Datasets
train_dl, val_dl, test_dl = load_dataset(
    dataset_name="live_fuel_moisture", base_path=BASE_PATH,
    batch_size=BATCH_SIZE, val_ratio=0.1
)

# Training Loop Acorss All Model
for model_name in MODELS:
    print(f"\n{'=' * 60}\nTraining model: {model_name.upper()}\n{'=' * 60}")

    wandb.init(
        project="live-fuel-moisture",
        name=model_name,
        config={
            "dataset":              DATASET_NAME,
            "model":                model_name,
            "lr":                   LR,
            "weight_decay":         WEIGHT_DECAY,
            "batch_size":           BATCH_SIZE,
            "n_epochs":             N_EPOCHS,
            "early_stop_patience":  EARLY_STOP,
        }
    )

    # Build a Model
    model = build_model(
        model_name=model_name,
        dataset_name=DATASET_NAME
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # schedular = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer = optimizer,
    #     mode = 'min',
    #     factor = 0.5,
    #     patience = 5
    #     )

    # Initiate the Trainer
    reg = RegressionTrainer(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            checkpoint_path = str(CHK_DIR / f"{DATASET_NAME}_{model_name}.pt"),
            std = 1.0,   # target is scaled → std ≈ 1
            use_wandb = True,
            # scheduler = schedular
            )
    # Training Starts here
    reg.fit(train_dl, val_dl, n_epochs=N_EPOCHS, early_stop_patience=EARLY_STOP)

    # Evaluation
    avg_loss, rmse_std = reg.evaluate(test_dl)
    results.append((model_name, avg_loss, rmse_std))

    wandb.finish()


# ─────────────────────────────────────────────
# Test Results Summary
# ─────────────────────────────────────────────
print(f"\n{'=' * 60}\nTest Results — LiveFuelMoistureContent\n{'=' * 60}")
print(f"{'Model':<8} | {'Test Loss':>10} | {'RMSE/std':>10}")
print("-" * 36)
for name, loss, rmse in results:
    print(f"{name.upper():<8} | {loss:>10.4f} | {rmse:>10.4f}")

best_name, best_loss, best_rmse = min(results, key=lambda r: r[2])
print("-" * 36)
print(f"Best model: {best_name.upper()} (RMSE/std: {best_rmse:.4f}, loss: {best_loss:.4f})")
