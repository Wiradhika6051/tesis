from abc import ABC, abstractmethod
from typing import List

from type import Sample


class CFGLocalizer(ABC):

    @abstractmethod
    def localize(
        self,
        cfg,
        sample: Sample
    ) -> List[int]:
        pass