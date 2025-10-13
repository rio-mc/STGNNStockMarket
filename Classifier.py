import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Classifier(nn.Module, ABC):
    """
    Abstract base class for all classifiers.

    Each subclass must implement:
        - forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Accepts a stock-tensor and returns a torch.Tensor of predictions.
        """
        pass