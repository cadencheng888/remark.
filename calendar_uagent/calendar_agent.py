"""
Calendar uAgent: wraps the existing, working calendar_tool.create_event so it's
reachable as a real Fetch.ai agent via the Chat Protocol — registrable on
Agentverse via Mailbox.

Run:
  export AGENT_SEED="<any unique secret string>"
  python3 calendar_agent.py
First run prints an "Agent inspector" URL — open it, click Connect -> Mailbox
to register on Agentverse. After that, the agent is discoverable/messageable
via Agentverse and ASI:One.

This intentionally reuses calendar_tool.py as-is (same code already proven
working via the FastAPI service) rather than reimplementing the Calendar API
call — lower risk under time pressure.
"""
import os
import re
from datetime import datetime, timedelta
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

import calendar_tool as cal

SEED = os.environ.get("AGENT_SEED", "calendar-agent-demo-seed-CHANGE-ME")

agent = Agent(
    name="calendar-action-agent",
    seed=SEED,
    port=8001,
    mailbox=True,
)

chat_proto = Protocol(spec=chat_protocol_spec)


def reply(text: str, end_session: bool = True) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


# Very small, deliberately narrow parser: "create <title> at <ISO time> [for N
# minutes]". Good enough to prove real agent-to-agent invocation works under
# time pressure — NOT meant to replace the LLM-based planner in calendar_action.ts.
CREATE_RE = re.compile(
    r"create\s+(?P<title>.+?)\s+at\s+(?P<start>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?)"
    r"(\s+for\s+(?P<duration>\d+)\s*min)?",
    re.IGNORECASE,
)


def handle_text(text: str) -> str:
    match = CREATE_RE.search(text)
    if not match:
        return (
            "I can create calendar events. Send a message like: "
            "'create Coffee with Sarah at 2026-06-22T14:00:00 for 30 min'"
        )
    title = match.group("title").strip()
    start_iso = match.group("start")
    duration = int(match.group("duration")) if match.group("duration") else 60
    try:
        result = cal.create_event(title=title, start_iso=start_iso, duration_minutes=duration)
        return result["message"]
    except Exception as e:
        return f"Failed to create event: {e}"


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Received from {sender}: {item.text}")
            response_text = handle_text(item.text)
            await ctx.send(sender, reply(response_text))


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(chat_proto, publish_manifest=True)

README = """# Calendar Action Agent

Creates Google Calendar events on request via natural-language chat messages.

## Usage
Send a chat message like:
  create Coffee with Sarah at 2026-06-22T14:00:00 for 30 min

The agent creates the event on the configured Google account's primary
calendar and replies with a confirmation link.
"""

if __name__ == "__main__":
    print(f"Agent address: {agent.address}")
    agent.run()