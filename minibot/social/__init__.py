"""Social layer — the robot's own accounts on outside services.

Structured like `robot/`: a transport that knows the wire protocol
(`x_client.py`) and a policy object above it that decides what is allowed to
go out (`account.py`). Nothing above this package knows that X speaks OAuth
1.0a, and nothing below it knows about drafts, rate limits or confirmation.
"""
