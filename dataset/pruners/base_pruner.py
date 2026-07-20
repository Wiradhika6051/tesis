from abc import ABC
from abc import abstractmethod


class BasePruner(ABC):

    @abstractmethod
    def prune(
        self,
        cfg,
        seed_nodes
    ):
        pass