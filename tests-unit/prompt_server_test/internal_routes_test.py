import pytest
from unittest.mock import patch, MagicMock
from api_server.routes.internal.internal_routes import InternalRoutes

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_prompt_server():
    server = MagicMock()
    return server

@pytest.fixture
def internal_routes_app(mock_prompt_server):
    internal_routes = InternalRoutes(mock_prompt_server)
    return internal_routes.get_app()

async def test_get_files_invalid_directory_type(aiohttp_client, internal_routes_app):
    client = await aiohttp_client(internal_routes_app)
    response = await client.get('/files/invalid_type')
    assert response.status == 400
    data = await response.json()
    assert data == {"error": "Invalid directory type"}

async def test_get_files_nonexistent_directory(aiohttp_client, internal_routes_app, tmp_path):
    non_existent_dir = str(tmp_path / "does_not_exist")
    with patch('api_server.routes.internal.internal_routes.get_directory_by_type', return_value=non_existent_dir):
        client = await aiohttp_client(internal_routes_app)
        response = await client.get('/files/input')
        assert response.status == 404
        data = await response.json()
        assert data == {"error": "Directory not found"}

async def test_get_files_valid_directory(aiohttp_client, internal_routes_app, tmp_path):
    valid_dir = tmp_path / "valid_input"
    valid_dir.mkdir()
    file1 = valid_dir / "test1.png"
    file1.write_text("dummy content 1")
    file2 = valid_dir / "test2.png"
    file2.write_text("dummy content 2")
    hidden_file = valid_dir / ".hidden"
    hidden_file.write_text("hidden")

    with patch('api_server.routes.internal.internal_routes.get_directory_by_type', return_value=str(valid_dir)):
        client = await aiohttp_client(internal_routes_app)
        response = await client.get('/files/input')
        assert response.status == 200
        data = await response.json()
        assert "test1.png" in data
        assert "test2.png" in data
        assert ".hidden" not in data
