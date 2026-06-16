from abc import ABC
from abc import abstractmethod

class DatasetStrategy(ABC):

    @abstractmethod
    def extract(self, file, repo):
        pass