"""X (Twitter) transport.

The only place in the application that knows X speaks HTTP + OAuth 1.0a. Same
shape as `robot/esp32_client.py`: one method per endpoint, retries with
backoff, serialized, and every failure surfaced as a typed exception rather
than a half-parsed response.

Auth is OAuth 1.0a user context, signed here with hmac/hashlib rather than
pulled in as a dependency. The robot posts as ONE fixed account and nobody
logs in interactively, so the four static credentials from the developer
portal are the whole story — OAuth 2.0 PKCE would add a browser redirect and a
refresh-token lifecycle to buy nothing. OAuth 1.0a also still signs the media
upload endpoints, which the app-only bearer token cannot do at all.

Credentials come from https://developer.x.com -> your app -> Keys and tokens.
The app must have **Read and write** permission, and the access token has to be
regenerated *after* that permission is set, or every post comes back 403.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass

import requests

from ..obs.logger import SOCIAL

API = "https://api.x.com"
# v1.1 upload host. Kept as a fallback because the v2 media endpoint is the
# newer of the two and older developer apps have been seen 403ing on it.
UPLOAD_V1 = "https://upload.twitter.com/1.1/media/upload.json"

# X counts a post's *weighted* length (URLs always count 23, CJK counts double),
# which cannot be reproduced exactly client-side. This is a cheap upper guard so
# obvious overruns never leave the Mac; the API is still the authority.
MAX_POST_CHARS = 280


class XUnavailable(RuntimeError):
    """X could not be reached after retries — network, DNS, timeout."""


class XApiError(RuntimeError):
    """X answered, and said no.

    `status` is the HTTP code and `detail` is X's own message, which is worth
    preserving verbatim: 403 "duplicate content", 403 "not permitted to perform
    this action" (read-only app) and 429 "usage cap exceeded" all look identical
    from the outside and need completely different fixes.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class XCredentials:
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""

    @property
    def complete(self) -> bool:
        return all((self.api_key, self.api_secret,
                    self.access_token, self.access_secret))


def _quote(value: str) -> str:
    """RFC 3986 percent-encoding. Python leaves `-._~` unreserved already, so
    only the default `safe="/"` has to be cleared."""
    return urllib.parse.quote(str(value), safe="")


class XClient:
    def __init__(self, creds: XCredentials, timeout: float = 15.0,
                 retries: int = 2):
        self.creds = creds
        self.timeout = timeout
        self.retries = retries
        self._s = requests.Session()
        # Posts are not idempotent and X rejects duplicates account-wide, so a
        # racing second caller would fail rather than double-post — but it would
        # also corrupt the rate-limit bookkeeping above. One at a time.
        self._lock = threading.RLock()
        self.stats = {"requests": 0, "failures": 0, "retries": 0}

    # -- signing ---------------------------------------------------
    def _auth_header(self, method: str, url: str,
                     query: dict[str, str] | None = None) -> str:
        """OAuth 1.0a HMAC-SHA1, per RFC 5849 §3.

        Only query parameters join the oauth_* parameters in the signature
        base. JSON and multipart bodies are deliberately excluded — the spec
        only folds in a body that is `application/x-www-form-urlencoded`, and
        signing one that isn't produces a valid-looking 401 that is miserable
        to debug. Every call below therefore puts signed parameters in the
        query string, never in a form body.
        """
        c = self.creds
        oauth = {
            "oauth_consumer_key": c.api_key,
            "oauth_nonce": secrets.token_hex(16),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": c.access_token,
            "oauth_version": "1.0",
        }
        params = {**oauth, **(query or {})}
        joined = "&".join(f"{_quote(k)}={_quote(params[k])}"
                          for k in sorted(params))
        base = f"{method.upper()}&{_quote(url)}&{_quote(joined)}"
        key = f"{_quote(c.api_secret)}&{_quote(c.access_secret)}".encode()
        digest = hmac.new(key, base.encode(), hashlib.sha1).digest()
        oauth["oauth_signature"] = base64.b64encode(digest).decode()

        inner = ", ".join(f'{_quote(k)}="{_quote(v)}"'
                          for k, v in sorted(oauth.items()))
        return f"OAuth {inner}"

    # -- core request ----------------------------------------------
    def _request(self, method: str, url: str, *, query: dict | None = None,
                 json: dict | None = None, files: dict | None = None,
                 timeout: float | None = None) -> dict:
        query = {k: str(v) for k, v in (query or {}).items()}
        timeout = timeout or self.timeout
        last: Exception | None = None

        with self._lock:
            for attempt in range(self.retries + 1):
                headers = {"Authorization": self._auth_header(method, url, query)}
                t0 = time.perf_counter()
                try:
                    r = self._s.request(method, url, params=query or None,
                                        json=json, files=files,
                                        headers=headers, timeout=timeout)
                except requests.RequestException as e:
                    last = e
                    self.stats["failures"] += 1
                    if attempt < self.retries:
                        self.stats["retries"] += 1
                        time.sleep(0.5 * (2 ** attempt))
                    continue

                self.stats["requests"] += 1
                ms = (time.perf_counter() - t0) * 1000
                path = urllib.parse.urlsplit(url).path
                SOCIAL.debug(f"{method} {path} -> {r.status_code} ({ms:.0f}ms)")

                if r.status_code < 400:
                    try:
                        return r.json()
                    except ValueError:
                        return {}

                detail = _explain(r)
                # 4xx is the account's own configuration or content — a retry
                # produces the identical answer. 429 is a real quota window
                # (the free tier's is measured in days), so retrying inside
                # this call would just burn the timeout.
                if r.status_code < 500:
                    self.stats["failures"] += 1
                    raise XApiError(r.status_code, detail)
                last = XApiError(r.status_code, detail)
                self.stats["failures"] += 1
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    time.sleep(0.5 * (2 ** attempt))

        if isinstance(last, XApiError):
            raise last
        raise XUnavailable(f"{method} {url}: {last}") from last

    # -- endpoints -------------------------------------------------
    def verify(self) -> dict:
        """GET /2/users/me — who the credentials actually belong to.

        Worth calling at startup: it is the only cheap way to tell a typo'd
        secret from a working one before the first real post.
        """
        data = self._request("GET", f"{API}/2/users/me").get("data", {})
        return {"id": data.get("id", ""), "username": data.get("username", ""),
                "name": data.get("name", "")}

    def upload_media(self, blob: bytes, mime: str = "image/jpeg") -> str:
        """Upload one image and return its media id.

        Tries the v2 endpoint first and falls back to v1.1, because which of
        the two a given developer app may use has moved more than once and the
        failure is a flat 403/404 rather than anything self-describing.
        """
        try:
            js = self._request(
                "POST", f"{API}/2/media/upload",
                query={"media_category": "tweet_image"},
                files={"media": ("frame.jpg", blob, mime)}, timeout=60)
            media_id = (js.get("data") or js).get("id") or js.get("media_id_string")
            if media_id:
                return str(media_id)
            raise XApiError(200, f"no media id in response: {js}")
        except XApiError as e:
            if e.status not in (403, 404, 405):
                raise
            SOCIAL.debug(f"v2 media upload unavailable ({e.status}), "
                         "falling back to v1.1")

        js = self._request("POST", UPLOAD_V1,
                           query={"media_category": "tweet_image"},
                           files={"media": ("frame.jpg", blob, mime)},
                           timeout=60)
        media_id = js.get("media_id_string") or (js.get("data") or {}).get("id")
        if not media_id:
            raise XApiError(200, f"no media id in response: {js}")
        return str(media_id)

    def post(self, text: str, media_ids: list[str] | None = None,
             reply_to: str | None = None) -> dict:
        """POST /2/tweets. Returns {"id", "text"}."""
        body: dict = {"text": text}
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        if reply_to:
            body["reply"] = {"in_reply_to_tweet_id": reply_to}
        data = self._request("POST", f"{API}/2/tweets", json=body).get("data", {})
        return {"id": data.get("id", ""), "text": data.get("text", text)}

    def mentions(self, user_id: str, limit: int = 5) -> list[dict]:
        """GET /2/users/:id/mentions — replies and @-mentions, newest first.

        Reads are metered far more tightly than writes; on the free tier this
        endpoint answers 429 outright. The caller is expected to let that
        surface as a plain "I can't check right now" rather than treat it as
        broken.
        """
        js = self._request(
            "GET", f"{API}/2/users/{user_id}/mentions",
            query={"max_results": max(5, min(100, int(limit))),
                   "tweet.fields": "created_at,author_id",
                   "expansions": "author_id", "user.fields": "username"})
        users = {u["id"]: u.get("username", "")
                 for u in (js.get("includes", {}).get("users") or [])}
        out = []
        for t in (js.get("data") or [])[:limit]:
            out.append({"id": t.get("id", ""), "text": t.get("text", ""),
                        "author": users.get(t.get("author_id", ""), ""),
                        "created_at": t.get("created_at", "")})
        return out


def _explain(r: requests.Response) -> str:
    """X's error shape is inconsistent across API versions — v2 uses `detail`
    or an `errors` list, v1.1 uses `errors[].message`. Pull out whichever is
    present so the model and the log get the real reason."""
    try:
        js = r.json()
    except ValueError:
        return (r.text or "").strip()[:200] or r.reason
    if isinstance(js, dict):
        if js.get("detail"):
            return str(js["detail"])
        errors = js.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                return str(first.get("message") or first.get("detail") or first)
            return str(first)
        if js.get("title"):
            return str(js["title"])
    return str(js)[:200]
