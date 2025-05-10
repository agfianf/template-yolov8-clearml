import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


path_to_env = os.path.join(os.getcwd(), ".env")
print("path_to_env", path_to_env, os.path.exists(path_to_env))


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
print("Environment variables loaded successfully.")
print("CVAT_HOST:", settings.CVAT_HOST)
