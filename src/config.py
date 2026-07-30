import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logging import get_logger


logger = get_logger(__name__)

path_to_env = os.path.join(os.getcwd(), ".env")
logger.debug("path_to_env %s exists=%s", path_to_env, os.path.exists(path_to_env))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=path_to_env,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    CVAT_USERNAME: str
    CVAT_PASSWORD: SecretStr
    CVAT_ORGANIZATION: str
    CVAT_FORMAT_DATA: str
    CVAT_HOST: str
    CVAT_OUTPUT_DIR: str


settings = Settings()
settings.CVAT_HOST = settings.CVAT_HOST.replace("\\x3a", ":")

TMP_DIR_CVAT = "./tmp-cvat"
# Never log CVAT_HOST: the console of every task is readable to anyone with
# access to the ClearML project, and this repository is public.
logger.debug("environment variables loaded successfully")
