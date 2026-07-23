from .recurrent_dataset import RecurrentDataset

try:
    from .stgnn_dataset import STGNNDataset
except ImportError:
    STGNNDataset = None

__all__ = [
    "RecurrentDataset",
    "STGNNDataset",
]
