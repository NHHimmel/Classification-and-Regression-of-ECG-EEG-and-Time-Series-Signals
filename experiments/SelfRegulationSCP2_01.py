import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATASET_CONFIGS, MODEL_CONFIGS
from utils.data_loader import load_dataset
from utils.model_factory import build_model
from utils.trainer import ClassificationTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb

BASE_PATH = "C:/Users/AlphaNHH/Downloads/Time Series Data/"
CHK_DIR = Path("./models")
CHK_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["cnn", 'rnn', 'lstm', 'gru']

DATASET_NAME = "self_regulation_scp2"
CONFIG = DATASET_CONFIGS[DATASET_NAME]

LR            = 1e-3
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 32
N_EPOCHS      = 100
EARLY_STOP    = 15

results = []


# Loading Datasets
train_dl, val_dl, test_dl = load_dataset(
    dataset_name="self_regulation_scp2", base_path=BASE_PATH,
    batch_size=BATCH_SIZE, val_ratio=0.1
)

# Training Loop Acorss All Model
for model_name in MODELS:
    print(f"\n{'=' * 60}\nTraining model: {model_name.upper()}\n{'=' * 60}")

    wandb.init(
        project="self-regulation-scp2",
        name=model_name,
        config={
            "dataset":              DATASET_NAME,
            "model":                model_name,
            "lr":                   LR,
            "weight_decay":         WEIGHT_DECAY,
            "batch_size":           BATCH_SIZE,
            "n_epochs":             N_EPOCHS,
            "early_stop_patience":  EARLY_STOP,
            "scheduler":            "ReduceLROnPlateau",
        }
    )

    # Build a Model
    model = build_model(
        model_name=model_name,
        dataset_name=DATASET_NAME
    )

    class_weights = CONFIG.get('class_weights')
    criterion = nn.CrossEntropyLoss(torch.tensor(class_weights) if class_weights else None)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        factor = 0.5,
        patience = 5
        )

    # Initiate the Trainer
    clf = ClassificationTrainer(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            checkpoint_path = str(CHK_DIR / f"{DATASET_NAME}_{model_name}.pt"),
            scheduler = schedular,
            use_wandb = True,
            )
    # Training Starts here
    clf.fit(train_dl, val_dl, n_epochs=N_EPOCHS, early_stop_patience=EARLY_STOP)

    # Evaluation
    avg_loss, accuracy = clf.evaluate(test_dl)
    results.append((model_name, avg_loss, accuracy))

    wandb.finish()


# ─────────────────────────────────────────────
# Test Results Summary
# ─────────────────────────────────────────────
print(f"\n{'=' * 60}\nTest Results — {DATASET_NAME}\n{'=' * 60}")
print(f"{'Model':<8} | {'Test Loss':>10} | {'Test Acc':>10}")
print("-" * 36)
for name, loss, acc in results:
    print(f"{name.upper():<8} | {loss:>10.4f} | {acc:>9.2f}%")

best_name, best_loss, best_acc = max(results, key=lambda r: r[2])
print("-" * 36)
print(f"Best model: {best_name.upper()} (acc: {best_acc:.2f}%, loss: {best_loss:.4f})")
