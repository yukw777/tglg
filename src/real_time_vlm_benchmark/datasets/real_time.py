import abc
from typing import Any, Callable

from torch.utils.data import Dataset


class RealTimeDataset(Dataset, abc.ABC):
    @property
    def preprocessor(self) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(
        self, preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None
    ) -> None:
        self._preprocessor = preprocessor

    @abc.abstractmethod
    def __len__(self) -> int:
        pass
