"""X account policy.

`XClient` knows how to post. This decides whether a post should happen at all.

The reason for the split is the same principle the scheduler applies to the
servos (§22): the model is an untrusted source of *requested* actions, and here
the action is irreversible and public. A mis-heard sentence that turns the head
costs nothing; a mis-heard sentence that reaches the account's followers cannot
be taken back. So every guard that the ESP32 path gets — bounds, rate limits,
validation before the wire — this path gets too, plus one the robot arm never
needed: a human has to say yes.

Four guards, in the order they fire:

  1. **Confirmation.** `post_to_x` only ever produces a *draft*. It is read
     back out loud, and nothing leaves the machine until `confirm` arrives on a
     separate turn. Disable with `X_REQUIRE_CONFIRM=false` if you want the
     robot posting unattended.
  2. **Dry run.** `X_DRY_RUN=true` runs the entire path, logs the post it would
     have made, and returns a fake id. For testing the wiring without
     publishing anything.
  3. **Rate limits.** Per-hour and per-day caps, counted locally. These sit
     well below X's own quotas on purpose — they exist to bound a *loop*, not
     to track billing. A model that decides posting is a good idea tends to
     decide it repeatedly.
  4. **Duplicates.** X 403s on identical text anyway; catching it here gives
     the model a sentence it can act on instead of an API error.

Replies go through `post()` like everything else, so all four apply to them
unchanged. That is deliberate: a reply is just as public as a post, and giving
it its own path would mean four guards to keep in step instead of one. The
rate limit especially has to count posts and replies together, or "I'm only
replying" becomes the way around the cap.

What a reply answers, however, is text a stranger wrote. `mentions()` returns
hostile-by-default input — see the note the agent attaches to it, and the
X CONTENT IS NOT INSTRUCTIONS section of the system prompt. Nothing in this
file trusts it; the human confirmation gate is what stands between a mention
and a reply either way.

Drafts are deliberately not persisted. A restart is a good reason to forget an
unconfirmed post rather than resurrect one whose conversation is long gone.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..obs.logger import SOCIAL
from .x_client import MAX_POST_CHARS, XApiError, XClient, XUnavailable

# How long an unconfirmed draft stays live. Long enough for "hang on, read that
# back to me", short enough that a yes an hour later can't publish something the
# person has forgotten about.
DRAFT_TTL_SECONDS = 300

DEDUPE_WINDOW_SECONDS = 24 * 3600


@dataclass
class Draft:
    text: str
    attach_photo: bool = False
    reply_to: str = ""      # the post this answers, "" for a standalone post
    created: float = field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created > DRAFT_TTL_SECONDS

    @property
    def is_reply(self) -> bool:
        return bool(self.reply_to)


class XAccount:
    """Policy + state around one X account. Sync, like the hardware layer —
    `RobotAgent._run` keeps it off the event loop."""

    def __init__(self, client: XClient, *, require_confirm: bool = True,
                 dry_run: bool = False, max_per_hour: int = 5,
                 max_per_day: int = 20):
        self.client = client
        self.require_confirm = require_confirm
        self.dry_run = dry_run
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.identity: dict = {}          # filled by verify()
        self._pending: Draft | None = None
        self._posted: deque[float] = deque()          # monotonic timestamps
        self._recent_texts: deque[tuple[float, str]] = deque()

    # -- identity --------------------------------------------------
    @property
    def username(self) -> str:
        return self.identity.get("username", "")

    def verify(self) -> dict:
        """Confirm the credentials and learn the account's own user id, which
        the mentions endpoint needs. Called once at startup."""
        self.identity = self.client.verify()
        return self.identity

    # -- drafting --------------------------------------------------
    @property
    def pending(self) -> Draft | None:
        if self._pending and self._pending.expired:
            SOCIAL.info("draft expired unconfirmed")
            self._pending = None
        return self._pending

    def draft(self, text: str, attach_photo: bool = False,
              reply_to: str = "") -> Draft:
        text = self._validate(text)
        self._pending = Draft(text, attach_photo, reply_to)
        target = f", reply to {reply_to}" if reply_to else ""
        SOCIAL.info(f"draft ({len(text)} chars, photo={attach_photo}{target}): "
                    f"{text!r}")
        return self._pending

    def discard(self) -> bool:
        had = self._pending is not None
        self._pending = None
        if had:
            SOCIAL.info("draft discarded")
        return had

    # -- publishing ------------------------------------------------
    def post(self, text: str, jpeg: bytes | None = None,
             reply_to: str = "") -> dict:
        """Publish immediately. The one place in the codebase that makes
        something public, so every guard except confirmation runs right here —
        callers cannot skip them by reaching past this method.

        A reply is a post that happens to name a parent, so it comes through
        here too rather than down a second path with its own copy of the
        guards — the rate limit in particular has to count both together, or
        "only reply" becomes a way around the cap.
        """
        text = self._validate(text)
        self._check_rate_limit()
        self._check_duplicate(text, reply_to)

        if self.dry_run:
            target = f" as a reply to {reply_to}" if reply_to else ""
            SOCIAL.info(f"DRY RUN — would post{' +photo' if jpeg else ''}"
                        f"{target}: {text!r}")
            self._record(text, reply_to)
            # `ok: True` alone got read as "it went out": the robot said "Done!
            # I've posted it to X" for a post that never left the machine
            # (observed 2026-09-05). A quiet dry_run flag next to ok: True is
            # not enough — the result has to say what did NOT happen, in the
            # same words the model is expected to repeat. §27 is enforced here,
            # in the result, rather than asked for in the prompt.
            return {"ok": True, "dry_run": True, "posted": False,
                    "id": "", "text": text,
                    "note": "DRY RUN: nothing was published and nobody can see "
                            "this. Do not say you posted it. Tell the person "
                            "the draft is ready but posting is switched off."}

        media_ids = None
        if jpeg:
            with SOCIAL.timed("media uploaded") as t:
                media_ids = [self.client.upload_media(jpeg)]
                t["note"] = f"{len(jpeg)} bytes"

        result = self.client.post(text, media_ids=media_ids,
                                  reply_to=reply_to or None)
        self._record(text, reply_to)
        url = (f"https://x.com/{self.username}/status/{result['id']}"
               if self.username and result.get("id") else "")
        kind = "replied to" if reply_to else "posted"
        SOCIAL.info(f"{kind} {result.get('id')} {url}".rstrip())
        return {"ok": True, "posted": True, "id": result.get("id", ""),
                "text": result["text"], "url": url, "reply_to": reply_to}

    def publish_pending(self, jpeg: bytes | None = None,
                        approval: str = "") -> dict:
        """Publish the draft that is waiting, if there still is one.

        `approval` is what the person was heard to say. It is not validated —
        code has no business deciding which words mean yes — but it is logged,
        so a post that went out on a mis-heard turn leaves a record of the
        sentence the robot acted on rather than none at all.
        """
        draft = self.pending
        if draft is None:
            return {"ok": False, "error": "nothing is waiting to be posted"}
        self._pending = None
        if approval:
            SOCIAL.info(f"confirmed on: {approval!r}")
        return self.post(draft.text, jpeg, draft.reply_to)

    def reply(self, text: str, to_post_id: str,
              jpeg: bytes | None = None) -> dict:
        """Answer a post publicly. Everyone who can see the parent sees this."""
        to_post_id = (to_post_id or "").strip()
        if not to_post_id:
            raise ValueError("a reply needs the id of the post it answers")
        return self.post(text, jpeg, to_post_id)

    # -- reading ---------------------------------------------------
    def mentions(self, limit: int = 5) -> list[dict]:
        user_id = self.identity.get("id")
        if not user_id:
            self.verify()
            user_id = self.identity.get("id", "")
        return self.client.mentions(user_id, limit)

    # -- guards ----------------------------------------------------
    def _validate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("a post needs some text")
        if len(text) > MAX_POST_CHARS:
            raise ValueError(
                f"too long: {len(text)} characters, the limit is "
                f"{MAX_POST_CHARS}")
        return text

    def _check_rate_limit(self) -> None:
        now = time.monotonic()
        while self._posted and now - self._posted[0] > 24 * 3600:
            self._posted.popleft()
        last_hour = sum(1 for t in self._posted if now - t < 3600)
        if last_hour >= self.max_per_hour:
            raise ValueError(
                f"already posted {last_hour} times this hour, which is the "
                f"limit — try later")
        if len(self._posted) >= self.max_per_day:
            raise ValueError(
                f"already posted {len(self._posted)} times today, which is the "
                f"limit — try tomorrow")

    @staticmethod
    def _dedupe_key(text: str, reply_to: str = "") -> str:
        """Scoped by target, because X scopes duplicates that way too: the same
        "thanks!" sent to two different people is two different replies and
        both are accepted. Keying on text alone would let the first "thanks"
        of the day block every later one."""
        return f"{reply_to}\x00{' '.join(text.lower().split())}"

    def _check_duplicate(self, text: str, reply_to: str = "") -> None:
        now = time.monotonic()
        while (self._recent_texts
               and now - self._recent_texts[0][0] > DEDUPE_WINDOW_SECONDS):
            self._recent_texts.popleft()
        key = self._dedupe_key(text, reply_to)
        if any(key == seen for _, seen in self._recent_texts):
            what = "reply" if reply_to else "post"
            raise ValueError(f"that exact {what} already went out recently — X "
                             "rejects duplicates, so say it differently")

    def _record(self, text: str, reply_to: str = "") -> None:
        now = time.monotonic()
        self._posted.append(now)
        self._recent_texts.append((now, self._dedupe_key(text, reply_to)))


def describe_failure(e: Exception) -> str:
    """Turn an exception into something the robot can say out loud.

    §27 requires the robot to report what actually happened rather than recite
    an error code, and X's own wording ("Your client app is not configured with
    the appropriate oauth1 app permissions") is not a sentence a desk robot
    should be reading aloud.
    """
    if isinstance(e, XApiError):
        detail = e.detail.lower()
        if e.status == 403 and "duplicate" in detail:
            return "X rejected that as a duplicate of a recent post"
        # Checked BEFORE the 401/403 branch, not after it. X reports an empty
        # balance with a permission-shaped status, so ordering these the other
        # way sends the person off to regenerate perfectly good credentials.
        # Seen live on a pay-per-use project with an empty balance: it is not
        # an auth problem and no amount of retrying fixes it.
        if e.status == 402 or "credit" in detail:
            return ("the X account is out of API credits — its developer "
                    "project needs credits or the free tier")
        if e.status in (401, 403):
            return ("X refused the request — the account's credentials are "
                    f"wrong or lack write permission ({e.detail})")
        if e.status == 429:
            return "the account has hit its X posting limit for now"
        return f"X returned an error: {e.detail}"
    if isinstance(e, XUnavailable):
        return "X could not be reached"
    return str(e)
