import io
import os
import tempfile
import zipfile
import pytest
from unittest.mock import patch, MagicMock

from app.frontend_management import download_release_asset_zip, Release


def create_mock_zip_bytes(files: dict[str, bytes]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, content in files.items():
            zf.writestr(file_path, content)
    return zip_buffer.getvalue()


def test_download_release_asset_zip_prevents_zip_slip():
    # Construct a zip with a malicious path traversal entry and a safe entry
    malicious_files = {
        "../evil.txt": b"malicious content",
        "index.html": b"<h1>Good Frontend</h1>",
    }
    zip_bytes = create_mock_zip_bytes(malicious_files)

    mock_release = Release(
        id=1,
        tag_name="1.0.0",
        name="Release 1.0.0",
        prerelease=False,
        created_at="2022-01-01T00:00:00Z",
        published_at="2022-01-01T00:00:00Z",
        body="Release notes",
        assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
    )

    mock_response = MagicMock()
    mock_response.content = zip_bytes
    mock_response.raise_for_status = MagicMock()

    with tempfile.TemporaryDirectory() as dest_dir:
        with patch("app.frontend_management.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Path traversal detected in zip archive entry"):
                download_release_asset_zip(mock_release, dest_dir)


def test_download_release_asset_zip_extracts_safe_zip():
    safe_files = {
        "index.html": b"<h1>Good Frontend</h1>",
        "assets/main.js": b"console.log('hello');",
    }
    zip_bytes = create_mock_zip_bytes(safe_files)

    mock_release = Release(
        id=1,
        tag_name="1.0.0",
        name="Release 1.0.0",
        prerelease=False,
        created_at="2022-01-01T00:00:00Z",
        published_at="2022-01-01T00:00:00Z",
        body="Release notes",
        assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
    )

    mock_response = MagicMock()
    mock_response.content = zip_bytes
    mock_response.raise_for_status = MagicMock()

    with tempfile.TemporaryDirectory() as dest_dir:
        with patch("app.frontend_management.requests.get", return_value=mock_response):
            download_release_asset_zip(mock_release, dest_dir)

        assert os.path.exists(os.path.join(dest_dir, "index.html"))
        assert os.path.exists(os.path.join(dest_dir, "assets", "main.js"))
        with open(os.path.join(dest_dir, "index.html"), "rb") as f:
            assert f.read() == b"<h1>Good Frontend</h1>"
