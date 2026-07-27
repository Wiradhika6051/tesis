from abc import ABC, abstractmethod
from typing import List

from src.type.Sample import Sample


class DiffLocalizer(ABC):

    @abstractmethod
    def localize(self, sample: Sample) -> List[int]:
        """
        Return changed source line numbers that should be used
        as pruning seeds.

        Parent (label=1):
            return removed (-) line numbers.

        Current (label=0):
            return added (+) line numbers.
        """
        pass