import io
import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from app.frontend_management import download_release_asset_zip, Release


def create_mock_zip(files: dict[str, bytes]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files.items():
            zip_file.writestr(filename, content)
    return zip_buffer.getvalue()


def test_download_release_asset_zip_path_traversal():
    malicious_zip_bytes = create_mock_zip({
        "safe.txt": b"safe content",
        "../traversal.txt": b"evil content",
    })

    release: Release = {
        "id": 1,
        "tag_name": "v1.0.0",
        "name": "v1.0.0",
        "prerelease": False,
        "created_at": "2025-01-01T00:00:00Z",
        "published_at": "2025-01-01T00:00:00Z",
        "body": "",
        "assets": [{"url": "https://example.com/dist.zip", "name": "dist.zip"}],
    }

    mock_response = MagicMock()
    mock_response.content = malicious_zip_bytes
    mock_response.raise_for_status.return_value = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Path traversal attempt detected"):
                download_release_asset_zip(release, tmp_dir)


def test_download_release_asset_zip_safe():
    safe_zip_bytes = create_mock_zip({
        "index.html": b"<html></html>",
        "sub/app.js": b"console.log('ok');",
    })

    release: Release = {
        "id": 1,
        "tag_name": "v1.0.0",
        "name": "v1.0.0",
        "prerelease": False,
        "created_at": "2025-01-01T00:00:00Z",
        "published_at": "2025-01-01T00:00:00Z",
        "body": "",
        "assets": [{"url": "https://example.com/dist.zip", "name": "dist.zip"}],
    }

    mock_response = MagicMock()
    mock_response.content = safe_zip_bytes
    mock_response.raise_for_status.return_value = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("requests.get", return_value=mock_response):
            download_release_asset_zip(release, tmp_dir)

        assert os.path.isfile(os.path.join(tmp_dir, "index.html"))
        assert os.path.isfile(os.path.join(tmp_dir, "sub", "app.js"))
