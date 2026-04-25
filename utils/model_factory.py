# utils/model_factory.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import DATASET_CONFIGS, MODEL_CONFIGS


# ─────────────────────────────────────────────
# Supported RNN cell types
# ─────────────────────────────────────────────

RNN_CELLS = {
    'rnn':  nn.RNN,
    'lstm': nn.LSTM,
    'gru':  nn.GRU,
}


# ─────────────────────────────────────────────
# Generic CNN
# ─────────────────────────────────────────────

class CNN1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: list,
                 pool_sizes: list, fc_sizes: list,
                 dropout_rate: float, cnn_kernels: list,
                 cnn_paddings: list, num_outputs: int):
        super().__init__()
        self.pool_sizes = pool_sizes

        # Conv blocks — one per entry in out_channels
        channels = [in_channels] + out_channels
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i+1], kernel_size=cnn_kernels[i], padding=cnn_paddings[i]),
                nn.BatchNorm1d(channels[i+1]),
            )
            for i in range(len(out_channels))
        ])
        

        # FC hidden layers
        fc_layers = []
        for size in fc_sizes:
            fc_layers.append(nn.LazyLinear(size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
        self.fc_hidden = nn.Sequential(*fc_layers)
        self.fc_out    = nn.LazyLinear(num_outputs)

    def forward(self, x):
        for block, pool_size in zip(self.conv_blocks, self.pool_sizes):
            x = torch.relu(F.avg_pool1d(block(x), kernel_size=pool_size, stride=pool_size))
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        return self.fc_out(x)


# ─────────────────────────────────────────────
# Generic plain RNN / LSTM / GRU
# ─────────────────────────────────────────────

class RNNModel(nn.Module):
    def __init__(self, cell_type: str, in_channels: int,
                 hidden_size: int, num_layers: int,
                 fc_sizes: list, num_outputs: int):
        super().__init__()
        self.is_lstm = (cell_type == 'lstm')

        self.rnn = RNN_CELLS[cell_type](
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

        # FC hidden layers
        fc_layers = []
        in_size = hidden_size
        for size in fc_sizes:
            fc_layers.append(nn.Linear(in_size, size))
            fc_layers.append(nn.ReLU())
            in_size = size
        self.fc_hidden = nn.Sequential(*fc_layers)
        self.fc_out    = nn.Linear(in_size, num_outputs)

    def forward(self, x):
        x = x.permute(2, 0, 1)            # (seq, batch, features)
        if self.is_lstm:
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)
        x = hn[-1]                         # last layer hidden state
        x = self.fc_hidden(x)
        return self.fc_out(x)


# ─────────────────────────────────────────────
# Generic RNN / LSTM / GRU with CNN frontend
# ─────────────────────────────────────────────

class RNNWithDS(nn.Module):
    def __init__(self, cell_type: str, in_channels: int,
                 ds_channels: list, ds_pool_sizes: list,
                 hidden_size: int, num_layers: int,
                 fc_sizes: list, cnn_kernels: list,
                 cnn_paddings: list, num_outputs: int):
        super().__init__()
        self.is_lstm = (cell_type == 'lstm')
        self.ds_pool_sizes = ds_pool_sizes

        # CNN downsampling
        channels = [in_channels] + ds_channels
        self.ds = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i+1], kernel_size=cnn_kernels[i], padding=cnn_paddings[i]),
                nn.BatchNorm1d(channels[i+1]),
            )
            for i in range(len(ds_channels))
        ])

        self.rnn = RNN_CELLS[cell_type](
            input_size=ds_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

        # FC hidden layers
        fc_layers = []
        in_size = hidden_size
        for size in fc_sizes:
            fc_layers.append(nn.Linear(in_size, size))
            fc_layers.append(nn.ReLU())
            in_size = size
        self.fc_hidden = nn.Sequential(*fc_layers)
        self.fc_out    = nn.Linear(in_size, num_outputs)

    def forward(self, x):
        for block, pool_size in zip(self.ds, self.ds_pool_sizes):
            x = torch.relu(F.max_pool1d(block(x), kernel_size=pool_size, stride=pool_size))
        x = x.permute(2, 0, 1)            # (seq, batch, features)
        if self.is_lstm:
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)
        x = hn[-1]
        x = self.fc_hidden(x)
        return self.fc_out(x)


# ─────────────────────────────────────────────
# Factory — single entry point
# ─────────────────────────────────────────────

def build_model(model_name: str, dataset_name: str) -> nn.Module:
    """
    Build and return a model instance.

    Args:
        model_name:   'cnn' | 'rnn' | 'lstm' | 'gru'
        dataset_name: 'abnormal_heartbeat' | 'self_regulation_scp2' |
                      'appliances_energy'  | 'live_fuel_moisture'

    Returns:
        Instantiated nn.Module
    """
    model_name   = model_name.lower()
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if model_name not in ('cnn', 'rnn', 'lstm', 'gru'):
        raise ValueError(f"Unknown model: {model_name}. Choose from: cnn, rnn, lstm, gru")

    dataset_cfg = DATASET_CONFIGS[dataset_name]
    model_cfg   = MODEL_CONFIGS[dataset_name]['cnn' if model_name=='cnn' else 'rnn']
    num_outputs = dataset_cfg['num_outputs']
    in_channels = dataset_cfg['input_shape'][0]

    if model_name == 'cnn':
        return CNN1d(
            in_channels  = in_channels,
            out_channels = model_cfg['out_channels'],
            pool_sizes   = model_cfg['pool_sizes'],
            fc_sizes     = model_cfg['fc_sizes'],
            dropout_rate = model_cfg['dropout_rate'],
            cnn_kernels  = model_cfg['cnn_kernels'],
            cnn_paddings = model_cfg['cnn_paddings'],
            num_outputs  = num_outputs,
        )

    else:  # rnn, lstm, gru
        if model_cfg['use_cnn_ds']:
            return RNNWithDS(
                cell_type            = model_name,
                in_channels          = in_channels,
                ds_channels    = model_cfg['ds_channels'],
                ds_pool_sizes  = model_cfg['ds_pool_sizes'],
                hidden_size          = model_cfg['hidden_size'],
                num_layers           = model_cfg['num_layers'],
                fc_sizes             = model_cfg['fc_sizes'],
                cnn_kernels          = model_cfg['cnn_kernels'],
                cnn_paddings         = model_cfg['cnn_paddings'],
                num_outputs          = num_outputs,
            )
        else:
            return RNNModel(
                cell_type   = model_name,
                in_channels = in_channels,
                hidden_size = model_cfg['hidden_size'],
                num_layers  = model_cfg['num_layers'],
                fc_sizes    = model_cfg['fc_sizes'],
                num_outputs = num_outputs,
            )