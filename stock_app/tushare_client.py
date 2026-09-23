"""Shared Tushare client configuration for the private proxy endpoint."""
from __future__ import annotations

import os

import tushare as ts
from dotenv import load_dotenv

load_dotenv()

FIXED_TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
TUSHARE_HTTP_URL = "http://lianghua.nanyangqiankun.top"


def get_tushare_token() -> str:
    """Return the platform token used by every data request."""
    return FIXED_TUSHARE_TOKEN


def configure_tushare_pro(pro, token: str | None = None):
    """Patch the private token/url fields required by the proxy-compatible API."""
    token = token or get_tushare_token()
    pro._DataApi__token = token
    pro._DataApi__http_url = TUSHARE_HTTP_URL
    return pro


def create_tushare_pro(timeout: int | None = None, token: str | None = None):
    """Create a Tushare pro client that always works with the private endpoint."""
    token = token or get_tushare_token()
    ts.set_token(token)
    if timeout is None:
        pro = ts.pro_api(token)
    else:
        pro = ts.pro_api(token, timeout=timeout)
    return configure_tushare_pro(pro, token)
