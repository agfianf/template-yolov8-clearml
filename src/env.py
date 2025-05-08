import os
from dotenv import load_dotenv
# load_dotenv('/workspace/src/.env')
load_dotenv()

CVAT_USERNAME = os.getenv("CVAT_USERNAME")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD")
CVAT_ORGANIZATION = os.getenv("CVAT_ORGANIZATION")
CVAT_FORMAT_DATA = os.getenv("CVAT_FORMAT_DATA")
CVAT_HOST = os.getenv("CVAT_HOST")
CVAT_OUTPUT_DIR = os.getenv("CVAT_OUTPUT_DIR")

TMP_DIR_CVAT = "./tmp-cvat"
print("load env")
print(CVAT_USERNAME)