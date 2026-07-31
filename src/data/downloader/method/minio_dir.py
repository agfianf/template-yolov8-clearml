import os
import time

from concurrent.futures import ThreadPoolExecutor

from minio import Minio

from src.utils.logging import get_logger


logger = get_logger(__name__)


class MinioDatasetDownloader:
    def __init__(self):
        self.__endpoint = os.getenv("MINIO_ENDPOINT")
        self.__access_key = os.getenv("MINIO_ACCESS_KEY")
        self.__secret_key = os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "app-data-workflow")
        self.region = os.getenv("MINIO_REGION", "xxxx-server-2")

        if not all([self.__endpoint, self.__access_key, self.__secret_key]):
            raise ValueError(
                "Missing required MinIO environment variables: "
                "MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
            )

        # Create a Minio client with the given credentials
        self.minio_client = Minio(
            self.__endpoint,
            access_key=self.__access_key,
            secret_key=self.__secret_key,
            secure=False,
            region=self.region,
        )

    def download_dataset(
        self, dataset_dict: dict, output_dir: str, max_workers: int = 10
    ) -> str:
        """params::
            - dataset_dict: {'class_name': ['url1', 'url2', ...], ...}
            - output_dir: path to save dataset

        Return:
            - list class name

        """
        # Create the download directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        ls_class = set()
        # Iterate over the classes in the dataset
        time_start = time.time()
        for class_name, urls in dataset_dict.items():
            # Create the class directory if it doesn't exist
            # 🚨 lowercase class_name
            class_name = class_name.lower()
            class_dir = os.path.join(
                output_dir, class_name
            )  # need_check_capital_class_name
            os.makedirs(class_dir, exist_ok=True)
            ls_class.add(class_name.capitalize())  # need_check_capital_class_name

            # Use a thread pool to download each file in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for url in urls:
                    # Extract the filename from the URL
                    filename = url.split("/")[-1]

                    # Construct the object name from the URL
                    object_name = url.split(self.bucket_name + "/")[1]

                    # Download the object to the class directory
                    destination_path = os.path.join(class_dir, filename)
                    executor.submit(
                        self.minio_client.fget_object,
                        self.bucket_name,
                        object_name,
                        destination_path,
                    )
        duration = round(time.time() - time_start, 2)
        logger.info(
            "minio: %d classes, %d files in %ss",
            len(ls_class),
            sum(len(urls) for urls in dataset_dict.values()),
            duration,
        )
        return ls_class


if __name__ == "__main__":
    # Placeholders on purpose: this repository is public, so a real endpoint or
    # key committed here is published. Point these at your own MinIO through the
    # environment rather than by editing them in.
    downloader = MinioDatasetDownloader(
        # endpoint=os.environ["MINIO_ENDPOINT"],       # noqa: ERA001  host:port
        # access_key=os.environ["MINIO_ACCESS_KEY"],   # noqa: ERA001
        # secret_key=os.environ["MINIO_SECRET_KEY"],   # noqa: ERA001
        # bucket_name="<bucket>",                      # noqa: ERA001
        dataset={
            "Empty": [
                "s3://<endpoint>/<bucket>/dataset/<project>/<split>/Empty/image_a.jpg",
                "s3://<endpoint>/<bucket>/dataset/<project>/<split>/Empty/image_b.jpg",
            ]
        },
        download_dir="./directory",
    )

    downloader.download_dataset(max_workers=10)
