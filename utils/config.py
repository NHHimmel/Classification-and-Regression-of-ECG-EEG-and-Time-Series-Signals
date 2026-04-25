DATASET_CONFIGS = {
    "abnormal_heartbeat": {
        "folder": "AbnormalHeartbeat",
        "task": "classification",
        "input_shape": (1, 18530),
        "num_classes": 2,
        'num_outputs': 2,
        "scale_features": False,
        "scale_target": False,
        "class_weights": [1.0, 2.71]  # 149 vs 55 → imbalanced
    },
    "self_regulation_scp2": {
        "folder": "SelfRegulationSCP2",
        "task": "classification",
        "input_shape": (7, 1152),
        "num_classes": 2,
        "num_outputs": 2,
        "scale_features": True,   # standard scaling needed
        "scale_target": False,
    },
    "appliances_energy": {
        "folder": "AppliancesEnergy",
        "task": "regression",
        "input_shape": (24, 144),
        "num_outputs": 1,
        "scale_features": True,
        "scale_target": True,     # your verdict
    },
    "live_fuel_moisture": {
        "folder": "LiveFuelMoistureContent",
        "task": "regression",
        "input_shape": (7, 365),
        "num_outputs": 1,
        "scale_features": True,   # normalization is a must
        "scale_target": True,
    },
}


MODEL_CONFIGS = {
    'abnormal_heartbeat': {
        'cnn': {
            'out_channels':  [4, 8, 8, 16],
            'pool_sizes':    [5, 5, 3],
            'fc_sizes':      [256, 128],
            'dropout_rate':  0.35,
            'cnn_kernels':[5, 5, 5, 5],
            'cnn_paddings':['same', 'same', 'same', 'same'],
        },
        'rnn': {
            'use_cnn_ds':     True,
            'ds_channels':    [4, 8],
            'ds_pool_sizes':  [5, 5],
            'hidden_size': 16,
            'num_layers': 3,
            'cnn_kernels': [7, 7],
            'cnn_paddings': ['valid', 'valid'],
            'fc_sizes': [],
        },
    },

    'self_regulation_scp2': {
        'cnn': {
            'out_channels':  [16, 32, 64],
            'pool_sizes':    [3, 3, 3],
            'fc_sizes':      [256, 128],
            'dropout_rate':  0.5,
            'cnn_kernels':[5, 5, 5],
            'cnn_paddings':['same', 'same', 'same'],
        },
        'rnn': {
            'use_cnn_ds':  False,
            'hidden_size': 256,
            'num_layers': 3,
            'fc_sizes': [128],
        },
    },

    'appliances_energy': {
        'cnn': {
            'out_channels':  [32, 64, 128],
            'pool_sizes':    [3, 3, 3],
            'fc_sizes':      [128, 64],
            'dropout_rate':  0.2,
            'cnn_kernels':[5, 5, 5],
            'cnn_paddings':['same', 'same', 'same'],
        },
        'rnn': {
            'use_cnn_ds':  False,
            'hidden_size': 128,
            'num_layers': 3,
            'fc_sizes': [128],
        },
    },

    'live_fuel_moisture': {
        'cnn': {
            'out_channels':  [16, 32, 64],
            'pool_sizes':    [3, 3, 3],
            'fc_sizes':      [128, 64],
            'dropout_rate':  0.2,
            'cnn_kernels':[5, 5, 5],
            'cnn_paddings':['same', 'same', 'same'],
        },
        'rnn': {
            'use_cnn_ds':  False,
            'hidden_size': 128,
            'num_layers': 3,
            'fc_sizes': [128],
        },
    },
}