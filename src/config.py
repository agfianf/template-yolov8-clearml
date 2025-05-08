from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    CVAT_USERNAME: str | None = None
    CVAT_PASSWORD: SecretStr | None = None
    CVAT_ORGANIZATION: str | None = None
    CVAT_FORMAT_DATA: str | None = None
    CVAT_HOST: SecretStr | None = None
    CVAT_OUTPUT_DIR: str | None = None


settings = Settings()

TMP_DIR_CVAT = "./tmp-cvat"
print("load env")
print(settings.CVAT_USERNAME)
print(settings.CVAT_PASSWORD)
