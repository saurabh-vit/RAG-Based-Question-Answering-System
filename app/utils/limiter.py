from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single limiter instance for the whole app.
# Routes import this instance so rate limiting uses the same storage/handler wiring.
limiter = Limiter(key_func=get_remote_address)

