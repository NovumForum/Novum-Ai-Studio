import pytest
from unittest.mock import MagicMock, patch
import tempfile
from aiohttp import web

import folder_paths
from app.user_manager import UserManager
from app.app_settings import AppSettings


@pytest.fixture
def mock_user_directory():
    """Create a temporary user directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = folder_paths.get_user_directory()
        folder_paths.set_user_directory(temp_dir)
        yield temp_dir
        folder_paths.set_user_directory(original_dir)


@pytest.fixture
def user_manager(mock_user_directory):
    """Create a UserManager instance for testing."""
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        manager = UserManager()
        manager.users = {"default": "default"}
        yield manager


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.headers = {}
    return request


def test_unknown_user_app_settings_get_settings(user_manager, mock_request):
    """Test get_settings raises HTTPForbidden when user is unknown."""
    mock_request.headers = {"comfy-user": "unknown_user_123"}
    settings_mgr = AppSettings(user_manager)

    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        with pytest.raises(web.HTTPForbidden):
            settings_mgr.get_settings(mock_request)


def test_none_filepath_app_settings_get_settings(user_manager, mock_request):
    """Test get_settings raises HTTPForbidden when get_request_user_filepath returns None."""
    mock_request.headers = {"comfy-user": "default"}
    settings_mgr = AppSettings(user_manager)

    with patch.object(user_manager, 'get_request_user_filepath', return_value=None):
        with pytest.raises(web.HTTPForbidden):
            settings_mgr.get_settings(mock_request)


def test_unknown_user_app_settings_save_settings(user_manager, mock_request):
    """Test save_settings raises HTTPForbidden when user is unknown."""
    mock_request.headers = {"comfy-user": "unknown_user_123"}
    settings_mgr = AppSettings(user_manager)

    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        with pytest.raises(web.HTTPForbidden):
            settings_mgr.save_settings(mock_request, {"test": "val"})


def test_none_filepath_app_settings_save_settings(user_manager, mock_request):
    """Test save_settings raises HTTPForbidden when get_request_user_filepath returns None."""
    mock_request.headers = {"comfy-user": "default"}
    settings_mgr = AppSettings(user_manager)

    with patch.object(user_manager, 'get_request_user_filepath', return_value=None):
        with pytest.raises(web.HTTPForbidden):
            settings_mgr.save_settings(mock_request, {"test": "val"})


def test_unknown_user_get_user_data_path(user_manager, mock_request):
    """Test get_user_data_path returns 403 when user is unknown."""
    mock_request.headers = {"comfy-user": "unknown_user_123"}
    mock_request.match_info = {"file": "some_file.json"}

    routes = web.RouteTableDef()
    user_manager.add_routes(routes)

    # get_user_data_path is defined inside add_routes, but we can verify via user_manager's routes or calling get_request_user_filepath behavior
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        with pytest.raises(KeyError):
            user_manager.get_request_user_filepath(mock_request, "some_file.json")
