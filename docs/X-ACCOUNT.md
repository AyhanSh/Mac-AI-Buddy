# The robot's X account

Mini Bot can post to an X account of its own: text, optionally with a photo it
takes at the moment of posting. It can also read replies and mentions.

Unconfigured is the default. With no credentials in `.env` the tools are never
registered and the robot behaves exactly as it did before — it does not know it
has an account, so it cannot claim to have posted.

## Getting the credentials

The robot posts as one fixed account and nobody logs in interactively, so it
uses **OAuth 1.0a user context**: four static values, no browser redirect, no
refresh-token lifecycle. It also happens to be the only scheme that still signs
the media-upload endpoints.

1. Sign in to <https://developer.x.com> **as the account the robot should post
   as** — the tokens inherit whoever is logged in. Make the robot its own
   account first if you don't want it posting as you.
2. Create a project and an app.
3. In **User authentication settings**, set app permissions to
   **Read and write**. (Read-only is the default, and it fails at post time,
   not at setup time.)
4. Go to **Keys and tokens** and generate:
   - API Key and Secret → `X_API_KEY`, `X_API_SECRET`
   - Access Token and Secret → `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`

**The access token must be generated *after* the write permission is set.** A
token minted while the app was read-only keeps that permission forever and
returns 403 on every post. If you set the permission second, hit *Regenerate*.

Then:

```bash
python -m minibot --x-check
```

It authenticates, prints the account it resolved to, and posts nothing. Do this
before the first live run — it is the only cheap way to tell a typo'd secret
from a working one.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `X_API_KEY` / `X_API_SECRET` | — | App consumer credentials |
| `X_ACCESS_TOKEN` / `X_ACCESS_TOKEN_SECRET` | — | The account's own tokens |
| `X_REQUIRE_CONFIRM` | `true` | Post only after a spoken yes |
| `X_DRY_RUN` | `false` | Run the whole path, publish nothing |
| `X_MAX_POSTS_PER_HOUR` | `5` | Local guard |
| `X_MAX_POSTS_PER_DAY` | `20` | Local guard |

Start with `X_DRY_RUN=true`. The logs show exactly what would have gone out.
Dry run still needs working credentials — startup verifies them either way, and
without them there is no account and no tools at all.

## How a post happens

With confirmation on (the default), posting takes two turns and a human
in between:

```
you    "post something about how the day went"
bot    calls post_to_x  -> draft stored, nothing sent
bot    "How about: Sat on a desk all day. Reviews mixed."
you    "yeah, send it"
bot    calls confirm_x_post -> photo captured if the draft asked for one,
                               uploaded, post published
bot    "posted"
```

`cancel_x_post` throws the draft away. An unconfirmed draft expires by itself
after five minutes, so a "yes" to some later question cannot publish something
the conversation has moved past. Drafts are not persisted — a restart forgets
them.

A photo is captured at **confirm** time, not draft time, so the picture matches
what is actually in front of the robot when the post goes out. If the camera is
down, the text still goes out and the model is told the photo is missing, so it
says so rather than describing a picture nobody got.

Set `X_REQUIRE_CONFIRM=false` and `post_to_x` publishes immediately. That is the
right setting for a robot posting on a schedule and the wrong one for a robot
sitting in a room with a microphone.

## Why the guards exist

This is the only capability in the codebase that leaves the room. Everything
else the model can do is reversible — a wrong head angle is corrected by the
next command. A published post is not, and the model's input is a
transcription of whatever was said near a desk.

So the layering matches `robot/`: `x_client.py` knows the wire, `account.py`
decides what is allowed, and the same §22 posture applies — the model is an
untrusted source of *requested* actions, and requests are validated before they
reach the wire, not after.

Guards, in the order they fire:

| Guard | Stops |
|---|---|
| Confirmation | Anything the person did not actually agree to |
| Draft TTL (5 min) | A stale yes publishing a forgotten draft |
| Dry run | Everything, while you are testing |
| Rate limits (hour/day) | A model that decides posting is a good idea repeatedly |
| Duplicate window (24 h) | X's own 403, caught early with a usable message |
| Length check | A 400 that costs a round trip |

The persona (§27) carries the editorial half: post only when asked, never offer,
never post what was said in confidence or anything about someone who is not in
the room, read the draft back verbatim, and say "posted" only once the result
confirms it.

`describe_failure()` turns X's error strings into sentences a desk robot can
say. "Your client app is not configured with the appropriate oauth1 app
permissions" becomes "X refused the request — the account's credentials are
wrong or lack write permission."

## Tools the model sees

| Tool | Registered when |
|---|---|
| `post_to_x(text, attach_photo)` | Credentials verified at startup |
| `confirm_x_post()` | ...and `X_REQUIRE_CONFIRM=true` |
| `cancel_x_post()` | ...and `X_REQUIRE_CONFIRM=true` |
| `check_x_mentions(limit)` | Credentials verified at startup |

Bad credentials warn at startup and register **nothing** — a robot that thinks
it can post but cannot would tell you it posted.

## Known limits

- **Reads need a paid tier.** `check_x_mentions` is a normal call on Basic and
  up; on the free tier X answers 429 and the robot says it cannot check right
  now. Posting works on the free tier (500 posts/month at the time of writing).
- **Media upload has two endpoints.** The client tries `POST /2/media/upload`
  and falls back to the v1.1 host on 403/404/405, because which one a given
  developer app may use has moved more than once and the failure is not
  self-describing.
- **One image per post.** Threads, polls, quote posts, video and alt text are
  not wired up.
- **Rate limits are per process.** They reset when the robot restarts. They
  bound a runaway loop within a session; X's own quota is the real ceiling.
