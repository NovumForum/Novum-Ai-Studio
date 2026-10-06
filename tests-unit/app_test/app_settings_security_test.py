import pytest
from unittest.mock import MagicMock
from aiohttp import web
from app.app_settings import AppSettings


def test_app_settings_none_filepath_raises_forbidden():
    mock_user_manager = MagicMock()
    mock_user_manager.get_request_user_filepath.return_value = None

    app_settings = AppSettings(mock_user_manager)
    mock_request = MagicMock()

    with pytest.raises(web.HTTPForbidden):
        app_settings.get_settings(mock_request)

    with pytest.raises(web.HTTPForbidden):
        app_settings.save_settings(mock_request, {"test": "val"})


def test_app_settings_unknown_user_raises_unauthorized():
    mock_user_manager = MagicMock()
    mock_user_manager.get_request_user_filepath.side_effect = KeyError("Unknown user")

    app_settings = AppSettings(mock_user_manager)
    mock_request = MagicMock()

    with pytest.raises(web.HTTPUnauthorized):
        app_settings.get_settings(mock_request)

    with pytest.raises(web.HTTPUnauthorized):
        app_settings.save_settings(mock_request, {"test": "val"})
