import abc
from typing import Callable

import torch
from torch import nn


class BaselineModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def preprocess(self, datapoint: dict) -> dict[str, torch.Tensor]:
        pass

    @property
    @abc.abstractmethod
    def collate_fn(self) -> Callable[[list[dict]], dict]:
        pass

    @abc.abstractmethod
    def predict(
        self, batch: dict, use_offloaded_cache: bool = False, **gen_kwargs
    ) -> dict[int, list]:
        pass
