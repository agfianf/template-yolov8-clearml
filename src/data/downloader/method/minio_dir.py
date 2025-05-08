import os
from concurrent.futures import ThreadPoolExecutor
import time
from minio import Minio


class MinioDatasetDownloader:
    def __init__(self):
        __ENDPOINT = os.getenv("MINIO_ENDPOINT")
        __ACCESS_KEY = os.getenv("ACCESS_KEY")
        __SECRET_KEY = os.getenv("SECRET_KEY")
        __BUCKET_NAME_DATA = os.getenv("BUCKET_NAME_DATA")

        self.__endpoint = "10.8.0.66:9000"
        self.__access_key = "bs_server_1"
        self.__secret_key = "zNAYleEDeCnlzaXJsd7MvXnQhPmZehIA"
        self.bucket_name = "app-data-workflow"
        self.region = "xxxx-server-2"

        # Create a Minio client with the given credentials
        self.minio_client = Minio(
            self.__endpoint,
            access_key=self.__access_key,
            secret_key=self.__secret_key,
            secure=False,
            region=self.region,
        )

    def download_dataset(self, dataset_dict:dict, output_dir:str, max_workers:int=10) -> str:
        """
        params::
            - dataset_dict: {'class_name': ['url1', 'url2', ...], ...}
            - output_dir: path to save dataset

        return: 
            - list class name
        """
        # Create the download directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        ls_class = set()
        # Iterate over the classes in the dataset
        print('\t⚡ Downloading dataset')
        time_start = time.time()
        for class_name, urls in dataset_dict.items():
            # Create the class directory if it doesn't exist
            # 🚨 lowercase class_name
            class_name = class_name.lower()
            class_dir = os.path.join(output_dir, class_name)  # need_check_capital_class_name
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
        print(f'\t✅ Completed download dataset in {duration} secs!')
        return ls_class


if __name__ == "__main__":
    downloader = MinioDatasetDownloader(
        # endpoint="10.8.0.66:9000",
        # access_key="bs_server_1",
        # secret_key="zNAYleEDeCnlzaXJsd7MvXnQhPmZehIA",
        # bucket_name="app-data-workflow",
        dataset={
            "Empty": [
                "s3://10.8.0.66:9000/app-data-workflow/dataset/Bousteud/val-stiched-named-revised/maturity/Empty/day4_part2_set11_flip_20220718150758_8ac9e9e4af33482d82561083d33555ff.jpg",
                "s3://10.8.0.66:9000/app-data-workflow/dataset/Bousteud/val-stiched-named-revised/maturity/Empty/day1_set5_side1_20220715143154_a418890a9e5744418841051150b60315.jpg",
            ]
        },
        download_dir="./directory",
    )

    downloader.download_dataset(max_workers=10)
