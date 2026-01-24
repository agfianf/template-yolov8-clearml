import os
import threading
import time

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import boto3.session
import splitfolders

from clearml import config as config_clearml

from data.utils import mkdir_p


def download_object(s3_client, url_s3, save_dir):
    parts = url_s3.split("/")
    bucket_name = parts[2]
    fpfile_name = '/'.join(parts[3:])
    file_name = parts[-1]
    downloaded_path = os.path.join(save_dir, file_name)
    s3_client.download_file(
        bucket_name,
        fpfile_name,
        downloaded_path
    )
    return True

def download_files_from_s3_multithreading(s3_paths, aws_access_key_id, aws_secret_access_key, save_dir):
    # Create a session and use it to make our client
    session = boto3.session.Session()
    s3_client = session.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # Dispatch work tasks with our s3_client
    mkdir_p(save_dir)
    print('start')
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_key = {executor.submit(download_object, s3_client, url_s3, save_dir): url_s3 for url_s3 in s3_paths}

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception

class HandlerS3:
    def __init__(self, output_dir:str) -> None:
        self.aws_access_key_id = config_clearml.get_config_for_bucket('').key
        self.aws_secret_access_key = config_clearml.get_config_for_bucket('').secret
        self.output_dir = output_dir
        self.tmp_dir = 'tmp-classification'

    def download_object(self, s3_client, url_s3):
        bucket_name = url_s3.replace('s3://', '').split('/')[0]
        prefix = url_s3.replace(f's3://{bucket_name}/', '')

        parts = prefix.split("/")
        # bucket_name = parts[2]
        # fpfile_name = '/'.join(parts[3:])
        file_name = parts[-1]
        folder_class = parts[-2]
        mkdir_p(os.path.join(self.tmp_dir, folder_class))
        downloaded_path = os.path.join(self.tmp_dir, folder_class, file_name)
        s3_client.download_file(
            bucket_name,
            prefix,
            downloaded_path
        )
        return True

    def download_files_from_s3_multithreading(self, s3_paths):
        # Create a session and use it to make our client
        session = boto3.session.Session()
        s3_client = session.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

        # Dispatch work tasks with our s3_client
        mkdir_p(self.tmp_dir)
        print('start')
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_key = {executor.submit(self.download_object, s3_client, url_s3): url_s3 for url_s3 in s3_paths}

            for future in futures.as_completed(future_to_key):
                key = future_to_key[future]
                exception = future.exception()

                if not exception:
                    yield key, future.result()
                else:
                    yield key, exception

    def extract_list(self, dir_s3_path:str):
        bucket_name = dir_s3_path.replace('s3://', '').split('/')[0]
        prefix = dir_s3_path.replace(f's3://{bucket_name}/', '')
        print(bucket_name, prefix)
        s3_client = boto3.client('s3',
                            aws_access_key_id=self.aws_access_key_id,
                            aws_secret_access_key=self.aws_secret_access_key
                        )

        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        ls_urls = []
        for obj in objects['Contents']:
            ls_urls.append(os.path.join(f's3://{bucket_name}', obj['Key']))
        return ls_urls

    def export(self, dir_s3_path):
        # download from s3
        s3_paths = self.extract_list(dir_s3_path)
        start_time = time.time()
        for key, result in self.download_files_from_s3_multithreading(s3_paths):
            if result != True:
                print('FAILED:', key, result)
        end_time = time.time() - start_time

        # splitting
        splitfolders.ratio(
            self.tmp_dir,
            output=self.output_dir,
            seed=1337,
            ratio=(.8, .2),
            move=True
        )

        # write yaml for yolo
        cls_names = os.listdir(os.path.join(self.output_dir, 'train'))
        print(cls_names)
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write("train: train/\n")
            f.write("val: val/\n")
            f.write(f"nc: {len(cls_names)}\n")
            f.write(f"names: {cls_names}\n")
        print(f'Done Download in {round(end_time, 3)} secs for {len(s3_paths)} images')
        return self.output_dir

if __name__ == '__main__':
    handler = HandlerS3('test-output-classification')
    handler.export('s3://10.8.0.66:9000/')
