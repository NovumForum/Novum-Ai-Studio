import json
import asyncio
import pytest
import comfy.cli_args
comfy.cli_args.args.cpu = True

import app.frontend_management
import aiohttp
from aiohttp import web
from PIL import Image
from io import BytesIO


def create_dummy_png_bytes():
    im = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))
    buf = BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def server_app():
    from server import PromptServer
    loop = asyncio.get_event_loop()
    server = PromptServer(loop)
    server.add_routes()
    return server.app


@pytest.mark.asyncio
async def test_upload_mask_missing_original_ref(aiohttp_client, server_app):
    client = await aiohttp_client(server_app)
    png_bytes = create_dummy_png_bytes()

    data = aiohttp.FormData()
    data.add_field('image', png_bytes, filename='mask.png', content_type='image/png')

    resp = await client.post('/upload/mask', data=data)
    assert resp.status == 400


@pytest.mark.asyncio
async def test_upload_mask_malformed_json_original_ref(aiohttp_client, server_app):
    client = await aiohttp_client(server_app)
    png_bytes = create_dummy_png_bytes()

    data = aiohttp.FormData()
    data.add_field('image', png_bytes, filename='mask.png', content_type='image/png')
    data.add_field('original_ref', '{invalid_json}')

    resp = await client.post('/upload/mask', data=data)
    assert resp.status == 400


@pytest.mark.asyncio
async def test_upload_mask_non_dict_original_ref(aiohttp_client, server_app):
    client = await aiohttp_client(server_app)
    png_bytes = create_dummy_png_bytes()

    data = aiohttp.FormData()
    data.add_field('image', png_bytes, filename='mask.png', content_type='image/png')
    data.add_field('original_ref', json.dumps(["not", "a", "dict"]))

    resp = await client.post('/upload/mask', data=data)
    assert resp.status == 400


@pytest.mark.asyncio
async def test_upload_mask_missing_filename_in_original_ref(aiohttp_client, server_app):
    client = await aiohttp_client(server_app)
    png_bytes = create_dummy_png_bytes()

    data = aiohttp.FormData()
    data.add_field('image', png_bytes, filename='mask.png', content_type='image/png')
    data.add_field('original_ref', json.dumps({"type": "output"}))

    resp = await client.post('/upload/mask', data=data)
    assert resp.status == 400
