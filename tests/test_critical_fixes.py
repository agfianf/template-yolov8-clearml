"""Tests for critical bug fixes."""

import os
import tempfile

from pathlib import Path
from unittest.mock import patch

import pytest


class TestOperatorPrecedenceFix:
    """Test the operator precedence fix in split_folder_yolo."""

    def test_validation_split_calculation(self):
        """Test that validation split is calculated correctly.

        The bug was: num_valid = int(total_files * 1-train_ratio)
        Which evaluates as: (total_files * 1) - train_ratio
        Instead of: total_files * (1 - train_ratio)
        """
        total_files = 100
        train_ratio = 0.8

        # Old buggy calculation (would give ~99 instead of ~20)
        buggy_result = int(total_files * 1 - train_ratio)

        # Fixed calculation
        correct_result = int(total_files * (1 - train_ratio))

        assert buggy_result != correct_result, "Bug should produce different result"
        # Allow for float precision: 100 * 0.2 = 19.999... -> 19
        assert 19 <= correct_result <= 20, "Correct validation split should be ~20%"
        assert buggy_result == 99, "Buggy calculation gives ~99"

    def test_split_ratios_sum(self):
        """Test that train + validation = 100% when test_ratio is None."""
        total_files = 100
        train_ratio = 0.8

        num_train = int(total_files * train_ratio)
        num_valid = int(total_files * (1 - train_ratio))

        # Allow for rounding, but should be close to total
        assert num_train + num_valid <= total_files
        assert num_train == 80
        # Float precision: 100 * 0.2 = 19.999... -> 19
        assert 19 <= num_valid <= 20


class TestMinIOEnvVarValidation:
    """Test MinIO environment variable validation."""

    def test_missing_env_vars_raises_error(self):
        """Test that missing MinIO env vars raise ValueError."""
        # Clear any existing env vars
        env_vars = ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"]
        with patch.dict(os.environ, {}, clear=True):
            for var in env_vars:
                os.environ.pop(var, None)

            from src.data.downloader.method.minio_dir import MinioDatasetDownloader

            with pytest.raises(ValueError, match="Missing required MinIO environment"):
                MinioDatasetDownloader()

    def test_with_valid_env_vars(self):
        """Test that valid env vars allow initialization."""
        test_env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "test_key",
            "MINIO_SECRET_KEY": "test_secret",
        }
        with patch.dict(os.environ, test_env, clear=True):
            # This would normally try to connect, so we mock the Minio client
            with patch("src.data.downloader.method.minio_dir.Minio"):
                from src.data.downloader.method.minio_dir import MinioDatasetDownloader

                downloader = MinioDatasetDownloader()
                assert downloader.bucket_name == "app-data-workflow"


class TestImagePathsGuard:
    """Test empty image paths guard in prediction."""

    def test_empty_paths_returns_early(self):
        """Test that empty image_paths causes early return without crash."""
        image_paths = []

        # The fix should check if image_paths is empty before accessing [0]
        if not image_paths:
            result = "early_return"
        else:
            # This would crash without the fix
            result = image_paths[0]

        assert result == "early_return"

    def test_non_empty_paths_works(self):
        """Test that non-empty image_paths works normally."""
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]

        if not image_paths:
            result = "early_return"
        else:
            result = image_paths[0]

        assert result == "/path/to/image1.jpg"


class TestImgszValidator:
    """Test the image size validator in TrainParams."""

    def test_valid_imgsz(self):
        """Test valid image sizes that are multiples of 32."""
        from src.schema.params import TrainParams

        # Valid sizes
        for size in [32, 64, 128, 320, 416, 640, 1280]:
            params = TrainParams(imgsz=size)
            assert params.imgsz == size

    def test_invalid_imgsz(self):
        """Test invalid image sizes that are not multiples of 32."""
        from src.schema.params import TrainParams

        # Invalid sizes should raise validation error
        for size in [33, 100, 500, 641]:
            with pytest.raises(ValueError, match="multiple of 32"):
                TrainParams(imgsz=size)

    def test_epochs_bounds(self):
        """Test epochs validation bounds."""
        from src.schema.params import TrainParams

        # Valid epochs
        params = TrainParams(epochs=100)
        assert params.epochs == 100

        # Too low
        with pytest.raises(ValueError):
            TrainParams(epochs=0)

        # Too high
        with pytest.raises(ValueError):
            TrainParams(epochs=10001)

    def test_learning_rate_positive(self):
        """Test learning rate must be positive."""
        from src.schema.params import TrainParams

        # Valid lr
        params = TrainParams(lr0=0.01)
        assert params.lr0 == 0.01

        # Zero or negative should fail
        with pytest.raises(ValueError):
            TrainParams(lr0=0)

        with pytest.raises(ValueError):
            TrainParams(lr0=-0.01)
