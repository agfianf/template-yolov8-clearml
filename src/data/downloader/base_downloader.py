from abc import ABC, abstractmethod


class BaseDownloader(ABC):

    @abstractmethod
    def get_local_dataset_coco(self, **kwargs):
        pass
