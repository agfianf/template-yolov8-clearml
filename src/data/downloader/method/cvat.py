import os
import shutil
import time
import zipfile

from urllib.parse import urljoin

import requests

from cvat_sdk import make_client
from requests.auth import HTTPBasicAuth

from src.config import settings
from src.data.downloader.base_downloader import BaseDownloader
from src.utils.logging import get_logger


logger = get_logger(__name__)

# CVAT builds the export archive asynchronously and answers 202 until it is
# ready. Without a pause the loop re-asks as fast as the round trip allows,
# which is both a needless load on the server and an unbounded log.
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 120

# Enough of an error body to recognise the failure, not enough to flood the
# console -- CVAT answers errors with full HTML pages.
BODY_PREVIEW_CHARS = 200

STATUS_MESSAGES = {
    200: "200: Download of file started",
    201: "201: Output file is ready for downloading",
    202: "202: Exporting has been started",
    405: "405: Format is not available",
}


def describe_task_status(response) -> str:
    """Human-readable form of an export-status response."""
    return STATUS_MESSAGES.get(response.status_code, f"{response.status_code}: unknown")


class CVATHTTPDownloaderV1(BaseDownloader):
    def __init__(self):
        __url_cvat = settings.CVAT_HOST
        __username_cvat = settings.CVAT_USERNAME
        __password_cvat = settings.CVAT_PASSWORD.get_secret_value()
        __output_dir_tmp = settings.CVAT_OUTPUT_DIR
        __format_data = settings.CVAT_FORMAT_DATA

        if not all([__url_cvat, __username_cvat, __password_cvat]):
            raise Exception("CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD must be set")

        self.base_url = urljoin(__url_cvat, "/api/v1/")
        self.auth = HTTPBasicAuth(__username_cvat, __password_cvat)
        self.data_format = __format_data
        self.download_dir = __output_dir_tmp
        os.makedirs(self.download_dir, exist_ok=True)

    def get_about_server(self):
        response = requests.get(
            url=urljoin(self.base_url, "server/about"),
            auth=self.auth,
        )
        if response.status_code == 200:
            return True, response.json()
        return False, response.text

    def get_task_info(self, task_id: int):
        task_url = urljoin(self.base_url, f"tasks/{task_id}")
        response = requests.get(
            url=task_url,
            auth=self.auth,
        )
        return response.json()

    def get_project_info(self, task_info):
        project_id = task_info["project_id"]
        project_url = urljoin(self.base_url, f"projects/{project_id}")
        response = requests.get(
            url=project_url,
            auth=self.auth,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def save_file(response, file_name) -> float:
        """Write the archive to disk and return its size in MB.

        No progress bar: `response.content` has already pulled the whole body
        into memory by the time we get here, so a bar over `iter_content` would
        be timing a memcpy rather than a download.
        """
        total_size_mb = round(len(response.content) / 1024000, 2)
        with open(file_name, "wb") as f:
            f.write(response.content)
        return total_size_mb

    def extract_file(self, file_name, project_info, task_info):
        with zipfile.ZipFile(file_name, "r") as zip_ref:
            extract_dir = (
                f"{self.download_dir}/{project_info['name']}/{task_info['name']}"
            )
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_ref.extractall(extract_dir)
            logger.debug("extracted to %s", extract_dir)
        os.remove(file_name)
        return extract_dir

    def upload_dataset(self, task_id: int, file_name: str):
        text_response = {
            201: "Uploading has finished",
            202: "Uploading has been started",
            405: "Format is not available",
        }

        upload_url = f"{self.base_url}/tasks/{task_id}/annotations"
        with open(file_name, "rb") as f:
            files = [
                (
                    "annotation_file",
                    ("instances_default.json", f, "application/json"),
                )
            ]
        while True:
            params = {"format": "COCO 1.0"}
            response = requests.put(
                url=upload_url,
                auth=self.auth,
                files=files,
                params=params,
            )
            message = text_response.get(
                response.status_code, response.text[:BODY_PREVIEW_CHARS]
            )
            if response.status_code not in (201, 202):
                logger.error(
                    "cvat task %s annotation upload failed: %s %s",
                    task_id,
                    response.status_code,
                    message,
                )
                break

            response.raise_for_status()
            if response.status_code == 201:
                logger.info("cvat task %s: %s", task_id, message)
                break
            logger.debug("cvat task %s upload: %s", task_id, message)

    def download(self, task_id: int, annotations_only: bool = False):
        """return: task_info, project_info, file_name_zip."""
        if annotations_only:
            download_url = urljoin(
                self.base_url, f"tasks/{task_id}/annotations?format=COCO%201.0"
            )
        else:
            download_url = urljoin(
                self.base_url, f"tasks/{task_id}/dataset?format=COCO%201.0"
            )
        task_info = self.get_task_info(task_id=task_id)
        try:
            project_info = self.get_project_info(task_info=task_info)
        except Exception as e:
            # Never the URL: it carries CVAT_HOST and this repo is public.
            logger.error("cvat task %s: cannot read project info: %s", task_id, e)
            raise Exception(
                f"ERROR taskid: {task_id} | task_info response={task_info}"
            ) from e

        timeout_start = time.time()
        polls = 0
        while True:
            response = requests.get(
                url=download_url,
                auth=self.auth,
            )
            polls += 1
            logger.debug(
                "cvat task %s poll %s: %s", task_id, polls, describe_task_status(response)
            )

            response.raise_for_status()
            if response.status_code == 201:
                response = requests.get(
                    url=download_url + "&action=download",
                    auth=self.auth,
                    allow_redirects=True,
                )
                logger.debug(
                    "cvat task %s download: %s", task_id, describe_task_status(response)
                )
                file_name_zip = f"{self.download_dir}/{task_info['name']}.zip"
                size_mb = self.save_file(response, file_name_zip)
                logger.info(
                    'cvat task %s "%s": ready after %s polls / %.1fs, %s MB',
                    task_id,
                    task_info.get("name"),
                    polls,
                    time.time() - timeout_start,
                    size_mb,
                )
                break

            if time.time() - timeout_start > POLL_TIMEOUT_SECONDS:
                # raise, not break: `file_name_zip` is unbound on this path, so
                # breaking used to surface as an UnboundLocalError below and hide
                # the timeout entirely.
                raise TimeoutError(
                    f"cvat task {task_id}: export not ready after"
                    f" {POLL_TIMEOUT_SECONDS}s ({polls} polls)"
                )

            time.sleep(POLL_INTERVAL_SECONDS)
        return task_info, project_info, file_name_zip

    def get_local_dataset_coco(self, task_ids: list[int], annotations_only: bool = False):
        """Return: ls_path_dataset = [dataset_dir1, ...)]."""
        ls_path_dataset = []
        for task_id in task_ids:
            logger.debug("downloading cvat task %s", task_id)
            task_info, project_info, file_name_zip = self.download(
                task_id, annotations_only=annotations_only
            )
            dataset_dir = self.extract_file(
                file_name_zip, project_info=project_info, task_info=task_info
            )
            # images_dir = f"{dataset_dir}/images"  # noqa: ERA001
            # annotations_dir = f"{dataset_dir}/annotations"  # noqa: ERA001
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset


class CVATHTTPDownloaderV2(BaseDownloader):
    def __init__(self):
        __url_cvat = settings.CVAT_HOST
        __username_cvat = settings.CVAT_USERNAME
        __password_cvat = settings.CVAT_PASSWORD.get_secret_value()
        __output_dir_tmp = settings.CVAT_OUTPUT_DIR
        __organization = settings.CVAT_ORGANIZATION
        __format_data = settings.CVAT_FORMAT_DATA

        if not all([__url_cvat, __username_cvat, __password_cvat]):
            raise Exception("CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD must be set")

        self.base_url = urljoin(__url_cvat, "/api/")
        self.auth = HTTPBasicAuth(__username_cvat, __password_cvat)
        self.data_format = __format_data
        self.organization = __organization
        self.download_dir = __output_dir_tmp
        os.makedirs(self.download_dir, exist_ok=True)

    def get_about_server(self):
        response = requests.get(
            url=urljoin(self.base_url, "server/about"),
            auth=self.auth,
        )
        if response.status_code == 200:
            return True, response.json()
        return False, response.text

    def get_task_info(self, task_id: int):
        task_url = urljoin(self.base_url, f"tasks/{task_id}")
        response = requests.get(
            url=task_url, auth=self.auth, params={"org": self.organization}
        )
        return response.json()

    def get_project_info(self, task_info):
        project_id = task_info["project_id"]
        project_url = urljoin(self.base_url, f"projects/{project_id}")
        response = requests.get(
            url=project_url,
            auth=self.auth,
            params={"org": self.organization},
            headers={"Organization": self.organization},
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def save_file(response, file_name) -> float:
        """Write the archive to disk and return its size in MB."""
        total_size_mb = round(len(response.content) / 1024000, 2)
        with open(file_name, "wb") as f:
            f.write(response.content)
        return total_size_mb

    def extract_file(self, file_name, project_info, task_info):
        with zipfile.ZipFile(file_name, "r") as zip_ref:
            extract_dir = (
                f"{self.download_dir}/{project_info['name']}/{task_info['name']}"
            )
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_ref.extractall(extract_dir)
            logger.debug("extracted to %s", extract_dir)
        os.remove(file_name)
        return extract_dir

    def upload_dataset(self, task_id: int, file_name: str):
        text_response = {
            201: "Uploading has finished",
            202: "Uploading has been started",
            405: "Format is not available",
        }

        upload_url = f"{self.base_url}/tasks/{task_id}/annotations"
        with open(file_name, "rb") as f:
            files = [
                (
                    "annotation_file",
                    ("instances_default.json", f, "application/json"),
                )
            ]
        while True:
            params = {"format": "COCO 1.0"}
            response = requests.put(
                url=upload_url,
                auth=self.auth,
                files=files,
                params=params,
            )
            message = text_response.get(
                response.status_code, response.text[:BODY_PREVIEW_CHARS]
            )
            if response.status_code not in (201, 202):
                logger.error(
                    "cvat task %s annotation upload failed: %s %s",
                    task_id,
                    response.status_code,
                    message,
                )
                break

            response.raise_for_status()
            if response.status_code == 201:
                logger.info("cvat task %s: %s", task_id, message)
                break
            logger.debug("cvat task %s upload: %s", task_id, message)

    def download(self, task_id: int, annotations_only: bool = False):
        """return: task_info, project_info, file_name_zip."""
        if annotations_only:
            download_url = urljoin(
                self.base_url, f"tasks/{task_id}/annotations?format=COCO%201.0"
            )
        else:
            download_url = urljoin(
                self.base_url, f"tasks/{task_id}/dataset?format=COCO%201.0"
            )
        task_info = self.get_task_info(task_id=task_id)
        project_info = self.get_project_info(task_info=task_info)

        timeout_start = time.time()
        polls = 0
        while True:
            response = requests.get(
                url=download_url, auth=self.auth, params={"org": self.organization}
            )
            polls += 1
            logger.debug(
                "cvat task %s poll %s: %s", task_id, polls, describe_task_status(response)
            )

            response.raise_for_status()
            if response.status_code == 201:
                response = requests.get(
                    url=download_url + "&action=download",
                    auth=self.auth,
                    allow_redirects=True,
                )
                logger.debug(
                    "cvat task %s download: %s", task_id, describe_task_status(response)
                )
                file_name_zip = f"{self.download_dir}/{task_info['name']}.zip"
                size_mb = self.save_file(response, file_name_zip)
                logger.info(
                    'cvat task %s "%s": ready after %s polls / %.1fs, %s MB',
                    task_id,
                    task_info.get("name"),
                    polls,
                    time.time() - timeout_start,
                    size_mb,
                )
                break

            if time.time() - timeout_start > POLL_TIMEOUT_SECONDS:
                # raise, not break -- see the note on the V1 downloader.
                raise TimeoutError(
                    f"cvat task {task_id}: export not ready after"
                    f" {POLL_TIMEOUT_SECONDS}s ({polls} polls)"
                )

            time.sleep(POLL_INTERVAL_SECONDS)
        return task_info, project_info, file_name_zip

    def get_local_dataset_coco(self, task_ids: list[int], annotations_only: bool = False):
        """return: ls_path_dataset = [dataset_dir1, ...)]."""
        ls_path_dataset = []
        for task_id in task_ids:
            logger.debug("downloading cvat task %s", task_id)
            task_info, project_info, file_name_zip = self.download(
                task_id, annotations_only=annotations_only
            )
            dataset_dir = self.extract_file(
                file_name_zip, project_info=project_info, task_info=task_info
            )
            # images_dir = f"{dataset_dir}/images"  # noqa: ERA001
            # annotations_dir = f"{dataset_dir}/annotations"  # noqa: ERA001
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset


class CVATSDKDownloader(BaseDownloader):
    def __init__(self):
        __url_cvat = settings.CVAT_HOST
        __username_cvat = settings.CVAT_USERNAME
        __password_cvat = settings.CVAT_PASSWORD.get_secret_value()
        __output_dir_tmp = settings.CVAT_OUTPUT_DIR
        __organization = settings.CVAT_ORGANIZATION
        __format_data = settings.CVAT_FORMAT_DATA

        if not all([__url_cvat, __username_cvat, __password_cvat]):
            raise Exception("CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD must be set")

        self.url_cvat = __url_cvat
        self.host = (
            __url_cvat.split(":")[0].replace("http://", "").replace("https://", "")
        )
        self.port = __url_cvat.split(":")[1].replace("/", "")
        self.__username = __username_cvat
        self.__password = __password_cvat
        self.data_format = __format_data
        self.organization = __organization
        self.download_dir = __output_dir_tmp
        os.makedirs(self.download_dir, exist_ok=True)

    def export_cvat_task_to_coco_with_images(self, task_id):
        # Create a Client instance bound to a local server and
        # authenticate using basic auth
        with make_client(
            host=self.url_cvat, credentials=(self.__username, self.__password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            project = client.projects.retrieve(task.project_id)

            filename_zip = f"{task.name}-{task.id}.zip"
            export_path = os.path.join(self.download_dir, filename_zip)
            if os.path.exists(export_path):
                os.remove(export_path)
            started = time.time()
            task.export_dataset(
                include_images=True, format_name="COCO 1.0", filename=export_path
            )
            logger.info(
                'cvat task %s "%s": exported in %.1fs, %s MB',
                task_id,
                task.name,
                time.time() - started,
                round(os.path.getsize(export_path) / 1024000, 2),
            )

            os.makedirs(self.download_dir, exist_ok=True)
            with zipfile.ZipFile(export_path, "r") as zip_ref:
                extract_dir = f"{self.download_dir}/{project.name}/{task.name}"
                zip_ref.extractall(extract_dir)

            os.remove(export_path)

            # images_dir = f"{extract_dir}/images"  # noqa: ERA001
            # annotations_dir = f"{extract_dir}/annotations"  # noqa: ERA001
        return extract_dir

    def get_local_dataset_coco(self, task_ids: list[int], annotations_only: bool = False):  # noqa: ARG002
        """return: ls_path_dataset = [(images_dir, annotations_dir), ...)]."""
        ls_path_dataset = []
        for task_id in task_ids:
            dataset_dir = self.export_cvat_task_to_coco_with_images(task_id)
            ls_path_dataset.append(dataset_dir)
        return ls_path_dataset


if __name__ == "__main__":
    CVATHTTPDownloaderV1()
    CVATSDKDownloader()
