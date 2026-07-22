from abc import ABC, abstractmethod
from typing import List


class CFGLocalizer(ABC):

    @abstractmethod
    def localize(
        self,
        cfg,
        seed_lines: List[int]
    ) -> List[int]:
        pass