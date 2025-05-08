import os
import shutil
import time
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import zipfile
from urllib.parse import urljoin
from typing import List
from src.data.downloader.base_downloader import BaseDownloader
from cvat_sdk import make_client
from rich import print
import src.env as env


class CVATHTTPDownloaderV1(BaseDownloader):
    def __init__(self):
        __URL_CVAT = env.CVAT_HOST
        __USERNAME_CVAT = env.CVAT_USERNAME
        __PASSWORD_CVAT = env.CVAT_PASSWORD
        __OUTPUT_DIR_TMP = env.TMP_DIR_CVAT
        __FORMAT_DATA = os.getenv("CVAT_FORMAT_DATA")

        if __URL_CVAT is None or __USERNAME_CVAT is None or __PASSWORD_CVAT is None:
            raise Exception('CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD must be set')

        self.base_url = urljoin(__URL_CVAT, '/api/v1/')
        self.auth = HTTPBasicAuth(__USERNAME_CVAT, __PASSWORD_CVAT)
        self.data_format = __FORMAT_DATA
        self.download_dir = __OUTPUT_DIR_TMP
        os.makedirs(self.download_dir, exist_ok=True)

    def get_task_info(self, task_id: int):
        task_url = urljoin(self.base_url, f'tasks/{task_id}')
        response = requests.get(
            url=task_url, 
            auth=self.auth,
        )
        return response.json()

    def get_project_info(self, task_info):
        project_id = task_info['project_id']
        project_url = urljoin(self.base_url, f'projects/{project_id}')
        response = requests.get(
            url=project_url, 
            auth=self.auth,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def print_task_status(response):
        status_messages = {
            200: '200: Download of file started',
            201: '201: Output file is ready for downloading',
            202: '202: Exporting has been started',
            405: '405: Format is not available',
        }
        print(status_messages.get(response.status_code, 'Unknown status'))

    @staticmethod
    def save_file(response, file_name):
        total_size_mb = round(len(response.content)/1024000, 2)
        print('Downloading', file_name)
        print('TOTAL SIZE: ', total_size_mb, 'MB')
        with open(file_name, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024000), total=total_size_mb, ncols=72):
                f.write(chunk)
        print('Downloaded: ', total_size_mb, 'MB')

    def extract_file(self, file_name, project_info, task_info):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            extract_dir = f"{self.download_dir}/{project_info['name']}/{task_info['name']}"
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_ref.extractall(extract_dir)
            print('Extracted to:', extract_dir)
        os.remove(file_name)
        return extract_dir

    def upload_dataset(self, task_id: int, file_name: str):
        text_response ={
            201: 'Uploading has finished',
            202: 'Uploading has been started',
            405: 'Format is not available',
        }

        upload_url = f"{self.base_url}/tasks/{task_id}/annotations"
        files=[
             ('annotation_file',('instances_default.json',open(file_name,'rb'), 'application/json'))
        ]
        while True:
            params = {'format': 'COCO 1.0'}
            response = requests.put(
                url=upload_url, 
                auth=self.auth, 
                files=files,
                params=params,
            )
            print(response.status_code, text_response.get(response.status_code, response.text))
            if response.status_code != 201 and response.status_code != 202:
                print("[ERROR_UPDATE_CVAT_ANNOTATIONS]",response.status_code, text_response.get(response.status_code, response.text))
                break

            response.raise_for_status()
            if response.status_code == 201:
                print(response.status_code, text_response[response.status_code])
                # response = requests.get(url=download_url + '&action=download', auth=self.auth, allow_redirects=True)
                # self.print_task_status(response)
                break

    def download(self, task_id: int, annotations_only: bool = False):
        """
        return: task_info, project_info, file_name_zip
        """
        if annotations_only:
            download_url = urljoin(self.base_url, f"tasks/{task_id}/annotations?format=COCO%201.0")
        else:
            download_url = urljoin(self.base_url, f'tasks/{task_id}/dataset?format=COCO%201.0')
        task_info = self.get_task_info(task_id=task_id)
        try:
            project_info = self.get_project_info(task_info=task_info)
        except Exception as e:
            print(download_url, e)
            raise Exception(f"ERROR taskid: {task_id} | task_info response=", task_info, e)
            


        timeout_start = time.time()
        first_exporting_progress = False
        while True:
            response = requests.get(
                url=download_url, 
                auth=self.auth,
            )

            if response.status_code == 202 and first_exporting_progress:
                first_exporting_progress = True
                print('🍆', end='\r', flush=True)
            else:
                self.print_task_status(response)
            
            response.raise_for_status()
            if response.status_code == 201:
                print()
                response = requests.get(url=download_url + '&action=download', auth=self.auth, allow_redirects=True)
                self.print_task_status(response)
                file_name_zip = f"{self.download_dir}/{task_info['name']}.zip"
                self.save_file(response, file_name_zip)
                break

            if time.time() - timeout_start > 120:
                print('Timeout Download DATA from CVAT')
                break
        return task_info, project_info, file_name_zip
    
    def get_local_dataset_coco(self, task_ids: List[int], annotations_only: bool = False):
        """
        return: ls_path_dataset = [dataset_dir1, ...)]
        """
        
        ls_path_dataset = []
        for task_id in task_ids:
            print("Downloading task_id: ", task_id, " ...")
            task_info, project_info, file_name_zip = self.download(task_id, annotations_only=annotations_only)
            dataset_dir = self.extract_file(file_name_zip, project_info=project_info, task_info=task_info)
            # images_dir = f"{dataset_dir}/images"
            # annotations_dir = f"{dataset_dir}/annotations"
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset

class CVATHTTPDownloaderV2(BaseDownloader):
    def __init__(self):
        __URL_CVAT = env.CVAT_HOST
        __USERNAME_CVAT = env.CVAT_USERNAME
        __PASSWORD_CVAT = env.CVAT_PASSWORD
        __OUTPUT_DIR_TMP = env.TMP_DIR_CVAT
        __ORGANIZATION = env.CVAT_ORGANIZATION
        __FORMAT_DATA = env.CVAT_FORMAT_DATA

        if __URL_CVAT is None or __USERNAME_CVAT is None or __PASSWORD_CVAT is None:
            raise Exception('CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD must be set')

        self.base_url = urljoin(__URL_CVAT, '/api/')
        self.auth = HTTPBasicAuth(__USERNAME_CVAT, __PASSWORD_CVAT)
        self.data_format = __FORMAT_DATA
        self.organization = __ORGANIZATION
        self.download_dir = __OUTPUT_DIR_TMP
        os.makedirs(self.download_dir, exist_ok=True)

    def get_task_info(self, task_id: int):
        task_url = urljoin(self.base_url, f'tasks/{task_id}')
        response = requests.get(
            url=task_url, 
            auth=self.auth,
            params={'org': self.organization}
        )
        return response.json()

    def get_project_info(self, task_info):
        project_id = task_info['project_id']
        project_url = urljoin(self.base_url, f'projects/{project_id}')
        response = requests.get(
            url=project_url, 
            auth=self.auth,
            params={'org': self.organization},
            headers={'Organization': self.organization}
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def print_task_status(response):
        status_messages = {
            200: '200: Download of file started',
            201: '201: Output file is ready for downloading',
            202: '202: Exporting has been started',
            405: '405: Format is not available',
        }
        print(status_messages.get(response.status_code, 'Unknown status'))

    @staticmethod
    def save_file(response, file_name):
        total_size_mb = round(len(response.content)/1024000, 2)
        print('Downloading', file_name, ' | SIZE: ', total_size_mb, 'MB')
        with open(file_name, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024000), total=total_size_mb, ncols=72):
                f.write(chunk)

    def extract_file(self, file_name, project_info, task_info):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            extract_dir = f"{self.download_dir}/{project_info['name']}/{task_info['name']}"
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_ref.extractall(extract_dir)
            print('Extracted to:', extract_dir)
        os.remove(file_name)
        return extract_dir

    def upload_dataset(self, task_id: int, file_name: str):
        text_response ={
            201: 'Uploading has finished',
            202: 'Uploading has been started',
            405: 'Format is not available',
        }

        upload_url = f"{self.base_url}/tasks/{task_id}/annotations"
        files=[
             ('annotation_file',('instances_default.json',open(file_name,'rb'), 'application/json'))
        ]
        while True:
            params = {'format': 'COCO 1.0'}
            response = requests.put(
                url=upload_url, 
                auth=self.auth, 
                files=files,
                params=params,
            )
            print(response.status_code, text_response.get(response.status_code, response.text))
            if response.status_code != 201 and response.status_code != 202:
                print("[ERROR_UPDATE_CVAT_ANNOTATIONS]",response.status_code, text_response.get(response.status_code, response.text))
                break

            response.raise_for_status()
            if response.status_code == 201:
                print(response.status_code, text_response[response.status_code])
                # response = requests.get(url=download_url + '&action=download', auth=self.auth, allow_redirects=True)
                # self.print_task_status(response)
                break

    def download(self, task_id: int, annotations_only: bool = False):
        """
        return: task_info, project_info, file_name_zip
        """
        if annotations_only:
            download_url = urljoin(self.base_url, f"tasks/{task_id}/annotations?format=COCO%201.0")
        else:
            download_url = urljoin(self.base_url, f'tasks/{task_id}/dataset?format=COCO%201.0')
        task_info = self.get_task_info(task_id=task_id)
        project_info = self.get_project_info(task_info=task_info)

        timeout_start = time.time()
        first_exporting_progress = False
        while True:
            response = requests.get(
                url=download_url, 
                auth=self.auth,
                params={'org': self.organization}
            )

            if response.status_code == 202 and first_exporting_progress:
                first_exporting_progress = True
                print('🍆', end='\r', flush=True)
            else:
                self.print_task_status(response)
            
            response.raise_for_status()
            if response.status_code == 201:
                print()
                response = requests.get(url=download_url + '&action=download', auth=self.auth, allow_redirects=True)
                self.print_task_status(response)
                file_name_zip = f"{self.download_dir}/{task_info['name']}.zip"
                self.save_file(response, file_name_zip)
                break

            if time.time() - timeout_start > 120:
                print('Timeout Download DATA from CVAT')
                break
        return task_info, project_info, file_name_zip
    
    def get_local_dataset_coco(self, task_ids: List[int], annotations_only: bool = False):
        """
        return: ls_path_dataset = [dataset_dir1, ...)]
        """
        
        ls_path_dataset = []
        for task_id in task_ids:
            print("Downloading task_id: ", task_id, " ...")
            task_info, project_info, file_name_zip = self.download(task_id, annotations_only=annotations_only)
            dataset_dir = self.extract_file(file_name_zip, project_info=project_info, task_info=task_info)
            # images_dir = f"{dataset_dir}/images"
            # annotations_dir = f"{dataset_dir}/annotations"
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset

class CVATSDKDownloader(BaseDownloader):
    def __init__(self):
        __URL_CVAT = os.getenv('CVAT_SERVER_HOST')
        __USERNAME_CVAT = os.getenv('CVAT_SERVER_USERNAME')
        __PASSWORD_CVAT = os.getenv('CVAT_SERVER_PASSWORD')
        __OUTPUT_DIR_TMP = os.getenv('TMP_DIR_CVAT')
        __ORGANIZATION = os.getenv('CVAT_ORGANIZATION')
        __FORMAT_DATA = os.getenv('CVAT_SERVER_FORMAT_DATA', 'COCO 1.0')

        if __URL_CVAT is None:
            raise Exception('CVAT_HOST is not defined')
        if __USERNAME_CVAT is None:
            raise Exception('CVAT_USERNAME is not defined')
        if __PASSWORD_CVAT is None:
            raise Exception('CVAT_PASSWORD is not defined')

        self.url_cvat = __URL_CVAT
        self.host = __URL_CVAT.split(':')[0].replace('http://', '').replace('https://', '')
        self.port = __URL_CVAT.split(':')[1].replace('/', '')
        self.__username = __USERNAME_CVAT
        self.__password = __PASSWORD_CVAT
        self.data_format = __FORMAT_DATA
        self.organization = __ORGANIZATION
        self.download_dir = __OUTPUT_DIR_TMP
        os.makedirs(self.download_dir, exist_ok=True)

    def export_cvat_task_to_coco_with_images(self, task_id):
        # Create a Client instance bound to a local server and authenticate using basic auth
        with make_client(host=self.url_cvat, credentials=(self.__username, self.__password)) as client:
            task = client.tasks.retrieve(task_id)
            project = client.projects.retrieve(task.project_id)

            filename_zip = f'{task.name}-{task.id}.zip'
            export_path = os.path.join(self.download_dir, filename_zip)
            if os.path.exists(export_path):
                os.remove(export_path)
            print('exporting...')
            task.export_dataset(include_images=True, format_name='COCO 1.0', filename=export_path)
            print('exported')
            
            os.makedirs(self.download_dir, exist_ok=True)
            with zipfile.ZipFile(export_path, 'r') as zip_ref:
                extract_dir = f"{self.download_dir}/{project.name}/{task.name}"
                zip_ref.extractall(extract_dir)
            
            os.remove(export_path)
            
            # images_dir = f"{extract_dir}/images"
            # annotations_dir = f"{extract_dir}/annotations"
        return extract_dir

    def get_local_dataset_coco(self, task_ids: List[int], annotations_only: bool = False):
        """
        return: ls_path_dataset = [(images_dir, annotations_dir), ...)]
        """
        ls_path_dataset = []
        for task_id in task_ids:
            dataset_dir = self.export_cvat_task_to_coco_with_images(task_id)
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset
    

if __name__ == '__main__':
    CVATHTTPDownloaderV1()
    CVATSDKDownloader()