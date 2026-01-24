from enum import Enum

from data.downloader.aws_s3_downloader import AWSS3Downloader
from data.downloader.cvat_downloader import CVATDownloader
from data.downloader.label_studio_downloader import LabelStudioDownloader
from data.downloader.roboflow_downloader import RoboflowDownloader


class DownloaderType(Enum):
    CVAT = 'CVAT'
    AWS_S3 = 'AWS_S3'
    LABEL_STUDIO = 'label_studio'
    ROBOFLOW = 'roboflow'

class DownloaderFactory:
    _downloaders = {
        DownloaderType.CVAT: CVATDownloader,
        DownloaderType.AWS_S3: AWSS3Downloader,
        DownloaderType.LABEL_STUDIO: LabelStudioDownloader,
        DownloaderType.ROBOFLOW: RoboflowDownloader
    }

    @staticmethod
    def create_downloader(source:DownloaderType, *params):
        if source not in DownloaderFactory._downloaders:
            raise ValueError(f'Invalid source: {source}')
        return DownloaderFactory._downloaders[source](*params)

