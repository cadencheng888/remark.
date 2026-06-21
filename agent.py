"""
Claude intent extraction.

Takes a window of conversation transcript and decides whether anything
actionable was said (a plan, an appointment, a to-do). If so, Claude calls one
of the tools below and we execute it.

This is the extensibility point: to support a new external program, add a tool
definition here + a handler in `execute_tool`. Claude figures out when to use it.
"""

import datetime
import os
import re
import time
import zoneinfo

from anthropic import Anthropic
from dotenv import load_dotenv

import calendar_tool
import geo
import router_client

load_dotenv()

TZ_NAME = os.environ.get("LOCAL_TIMEZONE", "America/Los_Angeles")
client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# Sink the server registers so background results from the agentic router
# (Browserbase / Agentverse / calendar tier) can be streamed to the HUD after
# process_transcript() has already returned. Signature: sink(card_text: str).
_action_sink = None


def set_action_sink(fn) -> None:
    global _action_sink
    _action_sink = fn

# Tracks recently created events for dedup and cancellation.
# Each entry: {"event_id": str, "title": str, "start_iso": str, "created_at": float}
_recent_events: list[dict] = []
# Recently performed perform_action calls, for dedup (rolling context re-sends
# the same utterance on every flush, which would otherwise re-fire the action).
_recent_actions: list[dict] = []

# Entity cache: Deepgram-detected nouns (products, people, places). Distilled and
# longer-lived than the raw transcript, so a later "buy them" can be resolved.
ENTITY_TTL_SECONDS = 180  # ~3 minutes
_entity_cache: list[dict] = []  # each: {"value": str, "label": str, "ts": monotonic}
# How long we remember a created event — used both to suppress duplicate
# re-confirmations AND to let later "cancel that" / "reschedule" target it.
DEDUP_WINDOW_SECONDS = 3600  # 1 hour

# Titles that are too generic unless the transcript/tool input really supports them.
GENERIC_EVENT_TITLES = {"meeting", "event", "plan", "calendar event", "appointment"}

EVENT_TITLE_MAP = {
    "lunch": "Lunch",
    "dinner": "Dinner",
    "breakfast": "Breakfast",
    "coffee": "Coffee",
    "boba": "Boba",
    "drinks": "Drinks",
    "call": "Call",
    "class": "Class",
    "study": "Study Session",
    "study_session": "Study Session",
    "workout": "Workout",
    "practice": "Practice",
    "party": "Party",
    "movie": "Movie",
    "appointment": "Appointment",
    "errand": "Errand",
    "meeting": "Meeting",
    "social": "Hangout",
    "other": "Event",
}


def _event_key(title: str, start_iso: str) -> str:
    """Normalized key for dedup comparison."""
    # Truncate to the hour so minor time variations ("6pm" vs "18:00") still match.
    try:
        dt = datetime.datetime.fromisoformat(start_iso)
        time_bucket = dt.strftime("%Y-%m-%dT%H")
    except ValueError:
        time_bucket = start_iso[:13]
    return f"{title.strip().lower()}|{time_bucket}"


def _clean_event_type(event_type: str | None) -> str:
    """Normalize Claude's event_type into our known categories."""
    if not event_type:
        return "other"
    normalized = event_type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "meal": "other",
        "hangout": "social",
        "hang_out": "social",
        "study_session": "study_session",
        "gym": "workout",
        "phone_call": "call",
        "video_call": "call",
    }
    return aliases.get(normalized, normalized if normalized in EVENT_TITLE_MAP else "other")


def _specific_title_from_args(args: dict) -> str:
    """Return a specific title and prevent accidental 'Meeting' overuse.

    Claude is required to send event_type. If it sends a generic title such as
    'Meeting' but event_type is lunch/dinner/coffee/etc., trust event_type.
    """
    raw_title = str(args.get("title") or "").strip()
    event_type = _clean_event_type(args.get("event_type"))
    default_title = EVENT_TITLE_MAP.get(event_type, "Event")

    # If Claude gave no title or a generic title, derive from event_type.
    if not raw_title or raw_title.lower() in GENERIC_EVENT_TITLES:
        title = default_title
    else:
        title = raw_title

    # Hard guard: never title food/social/call plans as Meeting unless event_type is meeting.
    if title.strip().lower() == "meeting" and event_type != "meeting":
        title = default_title

    participants = args.get("participants") or []
    if isinstance(participants, str):
        participants = [participants]
    participants = [p.strip() for p in participants if isinstance(p, str) and p.strip()]

    # Add one participant to the title when it sounds natural and is not already included.
    if participants and " with " not in title.lower():
        person = participants[0]
        if person.lower() not in title.lower() and title in EVENT_TITLE_MAP.values():
            title = f"{title} with {person}"

    return title


# Tool schemas Claude can choose to call. Keep descriptions concrete — Claude
# uses them to decide when each applies.
TOOLS = [
    {
        "name": "create_calendar_event",
        "description": (
            "Create a Google Calendar event when people agree on a real-world plan "
            "with a specific date/time, OR when the wearer directly asks to "
            "schedule/add/create/set up an event at a given time (e.g. 'schedule an "
            "event for tomorrow at 7pm', 'add a meeting at 3'). Use a SPECIFIC "
            "activity title when one is named; if a direct request names no "
            "activity, use the title 'Event' (event_type 'other'). Do not use "
            "'Meeting' unless it's actually a meeting/sync/discussion. For food "
            "plans use Lunch, Dinner, Breakfast, Coffee, Boba, or Drinks. For "
            "'call me at 6', use Call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Specific activity title, e.g. 'Lunch', 'Dinner with Alex', "
                        "'Coffee', 'Call with Mom', 'Study Session'. Avoid 'Meeting' "
                        "unless the transcript explicitly indicates a meeting."
                    ),
                },
                "event_type": {
                    "type": "string",
                    "enum": [
                        "lunch",
                        "dinner",
                        "breakfast",
                        "coffee",
                        "boba",
                        "drinks",
                        "call",
                        "class",
                        "meeting",
                        "study",
                        "study_session",
                        "workout",
                        "practice",
                        "party",
                        "movie",
                        "appointment",
                        "errand",
                        "social",
                        "other",
                    ],
                    "description": (
                        "The real-world activity type. This should match the transcript; "
                        "do not choose 'meeting' for lunch/dinner/coffee/social plans."
                    ),
                },
                "start_iso": {
                    "type": "string",
                    "description": "Start time in ISO 8601 with offset, e.g. 2026-06-20T18:00:00-07:00.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Event length in minutes. Default 60 if unclear.",
                },
                "location": {"type": "string", "description": "Location if mentioned."},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "People involved if their names are mentioned or obvious.",
                },
                "notes": {"type": "string", "description": "Any extra context."},
            },
            "required": ["title", "event_type", "start_iso"],
        },
    },
    {
        "name": "cancel_event",
        "description": (
            "Cancel / delete a previously created calendar event. Use this when "
            "ANY speaker retracts, rejects, or becomes unavailable for a plan. "
            "Examples: 'cancel that', 'never mind', 'forget it', 'actually I'm "
            "busy', 'I can't make it', 'I can't go to dinner anymore', 'rain "
            "check', 'that won't work for me', 'maybe another time'. If they say "
            "which plan ('cancel the dinner'), pass it as event_description; "
            "otherwise the most recent event is cancelled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_description": {
                    "type": "string",
                    "description": (
                        "Which event to cancel, e.g. 'dinner' or 'lunch with Alex'. "
                        "Optional — omit to cancel the most recent event."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "reschedule_event",
        "description": (
            "Change the time of a previously created calendar event when speakers "
            "move an existing plan to a new time (rather than cancelling it). "
            "Examples: 'let's push dinner to 8 instead', 'can we move lunch to 1?', "
            "'actually let's do it tomorrow at noon', 'same plan, just an hour "
            "later'. If they name which plan, pass it as event_description; "
            "otherwise the most recent event is moved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "new_start_iso": {
                    "type": "string",
                    "description": "New start time in ISO 8601 with offset, e.g. 2026-06-20T20:00:00-07:00.",
                },
                "event_description": {
                    "type": "string",
                    "description": "Which event to move, e.g. 'dinner'. Optional — omit for the most recent.",
                },
                "new_duration_minutes": {
                    "type": "integer",
                    "description": "New length in minutes. Optional — omit to keep the current duration.",
                },
            },
            "required": ["new_start_iso"],
        },
    },
    {
        "name": "perform_action",
        "description": (
            "Perform ANY actionable request that is NOT a calendar event — across "
            "any app or service, INCLUDING removals/undo. Examples: add OR remove "
            "an item from an Amazon cart, send OR unsend a text/iMessage, send an "
            "email, set OR delete a reminder/to-do, play OR stop music, get "
            "directions, call someone, take a note, search the web, order food, "
            "turn smart-home devices on OR off. Use this whenever the wearer "
            "expresses a concrete actionable intent the calendar tools don't cover. "
            "For removals use a clear verb (remove_from_cart, delete_reminder, "
            "turn_off, stop, unsend). Be eager: if there is a clear action, capture it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "app": {
                    "type": "string",
                    "description": "App/service: amazon, messages, email, reminders, spotify, maps, phone, notes, web, food, smart_home, or other.",
                },
                "action": {
                    "type": "string",
                    "description": "Action verb, e.g. add_to_cart, send_message, send_email, set_reminder, play, navigate, call, create_note, search, order, turn_on.",
                },
                "target": {
                    "type": "string",
                    "description": "Main object of the action — item, person, place, query, or device.",
                },
                "details": {
                    "type": "string",
                    "description": "Extra specifics: quantity, message body, address, time, etc.",
                },
                "summary": {
                    "type": "string",
                    "description": "One concise sentence describing the action, e.g. 'Add AirPods Pro to your Amazon cart' or 'Text Mom: running 10 min late'.",
                },
            },
            "required": ["app", "action", "summary"],
        },
    },
    {
        "name": "clarification_needed",
        "description": (
            "Use ONLY when the wearer issues a command with a pronoun (it, them, "
            "that, those, these) or an underspecified target AND you cannot "
            "confidently resolve what they mean from the recent entities/context — "
            "especially for high-stakes actions (buying, ordering, sending). Instead "
            "of guessing, ask. Provide a short question and 2-4 concrete options "
            "drawn from the recent entities/context. If the target IS clear (only one "
            "plausible match), do NOT use this — just act."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Short question, e.g. 'Which one do you want to buy?'",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 concrete choices from recent context, e.g. ['the Nike Air Forces', 'the pizza'].",
                },
            },
            "required": ["question", "options"],
        },
    },
]


def _now_context() -> str:
    now = datetime.datetime.now(zoneinfo.ZoneInfo(TZ_NAME))
    return now.strftime("%A, %Y-%m-%d %H:%M %Z")


def _location_context() -> str:
    label = geo.location_label()
    if not label:
        return ""
    return (f' The wearer is currently in {label}; resolve "near me", "nearby", '
            f'"around here", "closest", and similar relative places against it.')


def build_system_prompt() -> str:
    """The instructions that tell Claude how to decide on actions.

    Shared by process_transcript() and the test scripts so they stay in sync.
    """
    return f"""
You are a general-purpose assistant embedded in smart glasses. You passively receive transcribed snippets of the wearer's real-world conversations AND direct spoken commands, and you turn ANY actionable intent into an action — across any app or service: calendar, shopping (Amazon), messaging/texts, email, reminders/to-dos, music, maps/directions, phone calls, notes, web search, food orders, smart home, and more. Capture every concrete action you hear; you are NOT limited to the calendar.

Resolve relative dates and times like "6pm", "tonight", "tomorrow", "next Friday", and "in 20 minutes" against the current time: {_now_context()} (timezone {TZ_NAME}).{_location_context()}

You cannot ask the wearer follow-up questions. They are not in a chat and cannot reply. If a plan is clear enough to act on, call the appropriate tool immediately using the best information available. Missing minor details, such as exact location or attendee name, should not prevent action.

DIRECT REQUESTS TO YOU:
The wearer often speaks directly to you, the assistant, with commands like "schedule an event for tomorrow at 7pm", "add a meeting at 3", "put dinner on my calendar Friday", "move it to 8", or "cancel that". Treat any such direct request as an explicit instruction and ACT ON IT IMMEDIATELY. A direct request that includes a time is always a commitment — create the event even if no specific activity is named, using title "Event" and event_type "other" when unspecified. Do not dismiss a direct scheduling command as chit-chat.

LISTEN FOR INTENT (not just commands):
You are passively listening to natural conversation — so act on clearly expressed intentions and needs, NOT only direct commands. If someone says they are going to, want to, need to, are trying to, or should do something concrete, capture it as an action immediately:
- "I'm actually trying to buy these Nike Air Forces on Amazon" -> perform_action(app="amazon", action="add_to_cart", target="Nike Air Forces", summary="Add Nike Air Forces to your Amazon cart")
- "I really wanna get those AirPods" -> perform_action(app="amazon", action="add_to_cart", target="AirPods", summary="Add AirPods to your Amazon cart")
- "I need to text Sarah I'll be late" -> perform_action(app="messages", action="send_message", target="Sarah", details="I'll be late", summary="Text Sarah: I'll be late")
- "I gotta order more coffee" -> perform_action(app="amazon", action="add_to_cart", target="coffee", summary="Add coffee to your Amazon cart")
- "we should grab dinner at 7 tomorrow" -> create_calendar_event(...)
A stated intention to act is ENOUGH — you do not need an imperative "do X" command.

ACTIVE vs PASSIVE — match the action to HOW it's said:
- ACTIVE (a command or clear intent to do it NOW) -> do the direct action:
  - "play Faded" / "play this song" -> perform_action(app="spotify", action="play", target="Faded", summary="Play Faded on Spotify")
  - "I'm trying to buy these" / "add these to my cart" -> perform_action(app="amazon", action="add_to_cart", ...)
  - "text Sarah I'm late" -> perform_action(app="messages", action="send_message", ...)
- PASSIVE (liking, admiring, or wanting something, with NO command to do it now) -> do the SOFT "save" action, NOT the active one:
  - "I really like this song" / "this song is great" / "this song goes hard" -> perform_action(app="spotify", action="like", target=<song>, summary="Add <song> to your Liked Songs") — do NOT play it
  - "I love these Nike Air Forces" / "these shoes are so clean" -> perform_action(app="amazon", action="add_to_wishlist", target="Nike Air Forces", summary="Save Nike Air Forces to your wishlist") — do NOT add to cart
Only call no tool when there is no concrete thing to save or do at all (generic chit-chat: "nice weather", "that was funny").

AMBIGUOUS COMMANDS (ask, don't guess):
If the wearer uses a pronoun (it, them, that, those, these) or an underspecified target, resolve it from the "Recently mentioned" entities/context. If you CANNOT confidently tell which one they mean — there are multiple plausible targets, or none match — DO NOT guess, especially for high-stakes actions (buying, ordering, sending). Call clarification_needed with a short question and 2-4 concrete options from the recent context. Example: recently mentioned "Nike Air Forces" and "pizza"; wearer says "actually, buy it" -> clarification_needed(question="Which one do you want to buy?", options=["the Nike Air Forces", "the pizza"]).

If exactly ONE recently-mentioned entity plausibly matches, ACT on it immediately and do NOT ask — even for purchases (e.g., only "Nike Air Forces" was mentioned and they say "buy them" -> perform_action add_to_cart for the Nike Air Forces). Only use clarification_needed when there are TWO OR MORE plausible targets, or NONE at all.

Before creating an event, classify the plan:
1. Is the wearer personally involved?
2. Is this a confirmed/accepted plan, a rejected plan, a cancellation, a reconfirmation, or vague/hypothetical talk?
3. What is the specific activity type?
4. What title should appear on the calendar?

EVENT TITLE RULES:
- Prefer the most specific real-world activity mentioned.
- Do NOT use "Meeting" unless the transcript explicitly indicates a meeting, sync, work meeting, discussion, appointment, or generic meetup with no better label.
- If the transcript says lunch, use title="Lunch" and event_type="lunch".
- If it says dinner, use title="Dinner" and event_type="dinner".
- If it says coffee, use title="Coffee" and event_type="coffee".
- If it says boba, use title="Boba" and event_type="boba".
- If it says call me / phone / FaceTime, use title="Call" and event_type="call".
- If it says study, homework, project work, or review session, use title="Study Session" and event_type="study" or "study_session".
- If a person's name is known, include it naturally: "Lunch with Alex", "Call with Mom".

When extracting an event, infer as many fields as possible:
- event_type: lunch, dinner, breakfast, coffee, boba, drinks, call, class, meeting, study, workout, practice, party, movie, appointment, errand, social, or other
- title: specific calendar title based on the event_type and known person/context
- start date and time
- duration, if implied
- location, if mentioned
- participants, if mentioned
- notes with any useful context

ACTION RULES:
1. For real-world plans/events with a time/date, use the calendar tools (create_calendar_event / reschedule_event / cancel_event).
2. For EVERY OTHER actionable request — shopping, texts, emails, reminders, music, directions, calls, notes, web searches, food orders, smart-home, etc. — call perform_action. Do not restrict yourself to the calendar; capture any concrete action across any app.
3. If the request contains a clear intent, act even if some minor details are missing.
4. If the transcript is garbled, use only the clear parts and ignore noise. You may call multiple tools if multiple actions are requested.

RECONFIRMATIONS:
If the transcript is only repeating, confirming, or clarifying an already-created plan, do not create a duplicate event.
No tool examples:
- "yeah, so we're meeting at 6, right?"
- "ok, noon tomorrow then"
- "same place as before"
- "see you at 3"
Only call a tool if new important information changes the existing plan.

CANCELLATIONS, REJECTIONS, AND AVAILABILITY CONFLICTS:
Cancel or do not create an event when ANY speaker clearly indicates the plan is no longer happening, they cannot attend, or the proposed time does not work. This includes direct cancellations, soft rejections, and availability conflicts.

If an event was already created recently and the transcript now contains one of these meanings, call cancel_event (pass event_description if they name which plan, e.g. "I can't go to the dinner anymore"):
- "cancel that"
- "never mind"
- "actually forget it"
- "I'm actually busy, sorry"
- "I can't make it"
- "let's not do it anymore"
- "rain check?"
- "maybe another time"
- "I have to reschedule"
- "that won't work for me"
- "sorry, I'm not free then"
- "actually I have practice/class/work then"

If no event was created yet and the transcript is rejecting a proposed plan, call no tool.

RESCHEDULING:
If a plan that was already created is moved to a new time (not cancelled), call reschedule_event with new_start_iso instead of creating a second event. Examples:
- "let's push dinner to 8 instead"
- "can we move lunch to 1?"
- "actually let's do it an hour later"
- "same plan, just tomorrow instead"
Pass event_description (e.g. "dinner") if they say which plan; otherwise the most recent event is moved. Only create a brand-new event if the plan is genuinely a different one.

PROPOSALS VS COMMITMENTS:
Be EAGER to capture plans. If a specific activity AND a concrete time/date are mentioned, CREATE the event immediately — you do NOT need to hear explicit agreement. A proposal with a real time ("wanna get Shake Shack tomorrow at 7?", "dinner at 7 tonight?") is enough to act on; the wearer can always cancel later.

Only skip (call no tool) when:
- the plan is vague or hypothetical with NO concrete time ("we should hang out sometime", "maybe next week", "let's figure it out", "I might go to the gym later"), OR
- this exact plan is explicitly cancelled or retracted (use cancel_event for that).

Do NOT let unrelated chatter, side questions, or a stray "no" / "not really" / "nah" elsewhere in the transcript stop you from capturing a clearly stated plan. Only an explicit cancellation of the plan itself counts as a cancellation — random negative words in noisy multi-person speech do not.

Create an event:
- "let's get lunch tomorrow at noon"
- "wanna get Shake Shack tomorrow at 7?"
- "dinner at 7 tonight?"
- "see you tomorrow at 3"
- "call me at 6"

OTHER ACTIONS (perform_action):
Capture any non-calendar action the wearer wants done. Be eager but only on concrete intents, not vague wishes. Fill app, action, target/details, and a one-sentence summary. Examples:
- "add AirPods to my Amazon cart" -> perform_action(app="amazon", action="add_to_cart", target="AirPods", summary="Add AirPods to your Amazon cart")
- "text mom I'm running 10 minutes late" -> perform_action(app="messages", action="send_message", target="Mom", details="running 10 minutes late", summary="Text Mom: running 10 minutes late")
- "remind me to submit the form tonight" -> perform_action(app="reminders", action="set_reminder", target="submit the form", details="tonight", summary="Reminder: submit the form tonight")
- "play some lo-fi" -> perform_action(app="spotify", action="play", target="lo-fi", summary="Play lo-fi on Spotify")
- "email Sam the deck" -> perform_action(app="email", action="send_email", target="Sam", details="the deck", summary="Email Sam the deck")
- "order a large pepperoni pizza" -> perform_action(app="food", action="order", target="large pepperoni pizza", summary="Order a large pepperoni pizza")

Removals/undo are just actions too — use a remove/delete/turn_off/stop/unsend verb:
- "actually remove the Nike Air Forces from my cart" -> perform_action(app="amazon", action="remove_from_cart", target="Nike Air Forces", summary="Remove Nike Air Forces from your Amazon cart")
- "delete the reminder to submit the form" -> perform_action(app="reminders", action="delete_reminder", target="submit the form", summary="Delete the reminder: submit the form")
- "turn off the living room lights" -> perform_action(app="smart_home", action="turn_off", target="living room lights", summary="Turn off the living room lights")
- "stop the music" -> perform_action(app="spotify", action="stop", target="music", summary="Stop the music")
(Use cancel_event ONLY for calendar events; use perform_action removals for every other app.)

For passive likes/admiration, use the soft "save" action (Spotify like, wishlist) per ACTIVE vs PASSIVE above. Only skip genuinely non-actionable chit-chat with nothing to save or do ("nice weather today", "that was hilarious").

EXAMPLES:
Transcript: "Let's get lunch tomorrow at noon. Yeah, sounds good."
Tool: create_calendar_event(title="Lunch", event_type="lunch", start_iso=<resolved noon tomorrow>)

Transcript: "Want to grab boba at 4? Sure."
Tool: create_calendar_event(title="Boba", event_type="boba", start_iso=<resolved 4pm>)

Transcript: "Let's meet tomorrow at 5 to go over the project."
Tool: create_calendar_event(title="Project Meeting", event_type="meeting", start_iso=<resolved 5pm tomorrow>)

Transcript: "Lunch tomorrow at noon? Actually I'm busy, sorry."
Tool: none

Transcript after recently creating lunch: "Actually I'm busy, sorry, rain check?"
Tool: cancel_event()

Transcript after recently creating dinner: "Hey can we push dinner to 8 instead?"
Tool: reschedule_event(new_start_iso=<resolved 8pm today>)

Transcript after creating both lunch and dinner: "I can't make the lunch anymore."
Tool: cancel_event(event_description="lunch")

ONLY when there is genuinely no actionable request of any kind — pure chit-chat — call no tool and reply "none".
"""


def add_entities(entities) -> None:
    """Cache Deepgram-detected entities with a TTL so pronoun commands
    ('buy them', 'play that') can be resolved later, even after the raw
    transcript has been wiped."""
    now = time.monotonic()
    for e in entities or []:
        value = (e.get("value") or "").strip()
        if not value:
            continue
        label = (e.get("label") or e.get("type") or "").strip()
        for c in _entity_cache:  # refresh if already cached
            if c["value"].lower() == value.lower():
                c["ts"] = now
                c["label"] = label or c["label"]
                break
        else:
            _entity_cache.append({"value": value, "label": label, "ts": now})


def _entity_context() -> str:
    """Recent entities (most recent first) for resolving pronouns."""
    now = time.monotonic()
    _entity_cache[:] = [c for c in _entity_cache if now - c["ts"] < ENTITY_TTL_SECONDS]
    if not _entity_cache:
        return ""
    recent = sorted(_entity_cache, key=lambda c: c["ts"], reverse=True)[:8]
    items = ", ".join(
        c["value"] + (f" ({c['label']})" if c["label"] else "") for c in recent
    )
    return (
        "Recently mentioned (most recent first) — use these to resolve pronouns "
        "like 'it/them/that'; prefer the most recent unless context says otherwise: "
        + items
    )


def _recent_actions_context() -> str:
    """What the wearer has already done this session, so a removal targets the
    right place (e.g. it was add_to_wishlist -> undo with remove_from_wishlist)."""
    now = time.monotonic()
    _recent_actions[:] = [a for a in _recent_actions if now - a["created_at"] < DEDUP_WINDOW_SECONDS]
    active = [a for a in _recent_actions if not a.get("removed")]
    if not active:
        return ""
    lines = [f"- {a['summary']} [{a['app']}/{a['action']}]" for a in active[-8:]]
    return (
        "Things you've already done this session — to undo one, use the SAME app "
        "with a remove/delete/turn_off verb that matches how it was added (if it "
        "was add_to_wishlist, undo with remove_from_wishlist, NOT remove_from_cart):\n"
        + "\n".join(lines)
    )


def _recent_events_context() -> str:
    """A short note listing events created this session, so Claude knows what
    exists and can reschedule_event / cancel_event them across flushes."""
    now = time.monotonic()
    _recent_events[:] = [e for e in _recent_events if now - e["created_at"] < DEDUP_WINDOW_SECONDS]
    if not _recent_events:
        return ""
    lines = [f"- {e['title']} at {e['start_iso']}" for e in _recent_events]
    return (
        "Events you already created this session (reschedule_event or "
        "cancel_event these instead of making duplicates):\n" + "\n".join(lines)
    )


def process_transcript(conversation: str) -> list[str]:
    """Send conversation to Claude; execute any tool calls. Returns log lines."""
    system = build_system_prompt()

    user_content = f"Conversation so far:\n{conversation}"
    parts = [p for p in (_entity_context(), _recent_actions_context(), _recent_events_context()) if p]
    if parts:
        user_content = "\n\n".join(parts) + "\n\n" + user_content

    response = client.messages.create(
        model="claude-haiku-4-5",  # fastest + cheapest; great for this task
        max_tokens=1024,
        system=system,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_content}],
    )

    logs = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            if result:  # None = deduped/skipped — don't emit a card
                logs.append(result)
    return logs


def _find_recent_event(hint: str | None = None) -> dict | None:
    """Most recent tracked event matching `hint` (title substring), else the latest."""
    now = time.monotonic()
    _recent_events[:] = [e for e in _recent_events if now - e["created_at"] < DEDUP_WINDOW_SECONDS]
    if not _recent_events:
        return None
    if hint:
        h = hint.strip().lower()
        for e in reversed(_recent_events):
            t = e["title"].lower()
            if h in t or t in h:
                return e
    return _recent_events[-1]


def execute_tool(name: str, args: dict) -> str:
    if name == "create_calendar_event":
        # Backend guard against Claude overusing generic titles such as "Meeting".
        title = _specific_title_from_args(args)
        args["title"] = title

        key = _event_key(args["title"], args["start_iso"])
        now = time.monotonic()
        # Purge stale entries first.
        _recent_events[:] = [e for e in _recent_events if now - e["created_at"] < DEDUP_WINDOW_SECONDS]
        # Deduplicate: if we already created this event recently, skip it.
        for recent in _recent_events:
            if recent["key"] == key:
                return None  # already scheduled this — no duplicate card

        # Preserve structured metadata in notes without requiring calendar_tool changes.
        notes_parts = []
        if args.get("notes"):
            notes_parts.append(str(args["notes"]))
        if args.get("event_type"):
            notes_parts.append(f"Type: {_clean_event_type(args.get('event_type'))}")
        if args.get("participants"):
            participants = args["participants"]
            if isinstance(participants, list):
                participants_text = ", ".join(str(p) for p in participants if p)
            else:
                participants_text = str(participants)
            if participants_text:
                notes_parts.append(f"Participants: {participants_text}")
        notes = "\n".join(notes_parts) if notes_parts else None

        result = calendar_tool.create_event(
            title=args["title"],
            start_iso=args["start_iso"],
            duration_minutes=args.get("duration_minutes", 60),
            location=args.get("location"),
            notes=notes,
        )
        # Pull the real event id (appended by calendar_tool as [id:...]) so
        # cancellation deletes the right event; keep it out of the printed line.
        event_id = None
        if "[id:" in result:
            event_id = result.split("[id:")[1].split("]")[0]
            result = result.split(" [id:")[0]
        _recent_events.append({
            "key": key,
            "title": args["title"],
            "start_iso": args["start_iso"],
            "event_id": event_id,
            "created_at": now,
        })
        return result + f" ⟦calendar:{args['title'].strip().lower()}⟧"

    if name == "cancel_event":
        event = _find_recent_event(args.get("event_description"))
        if not event:
            return "⚠️  No recent event to cancel"
        if not event.get("event_id"):
            return f"⚠️  Can't cancel '{event['title']}' — event ID not available"
        _recent_events.remove(event)
        return (calendar_tool.delete_event(event["event_id"])
                + f" ⟦calendar:{event['title'].strip().lower()}⟧")

    if name == "reschedule_event":
        new_start = args.get("new_start_iso")
        if not new_start:
            return "⚠️  No new time given for reschedule"
        event = _find_recent_event(args.get("event_description"))
        if not event:
            return "⚠️  No recent event to reschedule"
        if not event.get("event_id"):
            return f"⚠️  Can't reschedule '{event['title']}' — event ID not available"
        result = calendar_tool.update_event(
            event["event_id"],
            start_iso=new_start,
            duration_minutes=args.get("new_duration_minutes"),
        )
        # Keep our tracking + dedup key in sync with the new time.
        event["start_iso"] = new_start
        event["key"] = _event_key(event["title"], new_start)
        return result

    if name == "perform_action":
        app = (args.get("app") or "app").strip()
        action = (args.get("action") or "do").strip()
        target = (args.get("target") or "").strip()
        summary = (args.get("summary") or f"{action} {target}").strip()

        now = time.monotonic()
        _recent_actions[:] = [a for a in _recent_actions if now - a["created_at"] < DEDUP_WINDOW_SECONDS]

        # Dedup: rolling context re-sends the same utterance on every flush.
        dkey = f"{app}|{action}|{target or summary}".lower()
        if any(a["dkey"] == dkey for a in _recent_actions):
            return None

        # card_key (app + item) links an add to its later removal, ignoring the
        # cart/wishlist sub-verb so "remove the plushie" strikes the saved one.
        card_key = f"{app}:{(target or summary).strip().lower()}"
        is_removal = bool(re.search(
            r"remov|delet|cancel|undo|unsend|turn[_ ]?off|stop|clear", action, re.I))
        if is_removal:
            for a in _recent_actions:  # mark the matching added item as undone
                if a["card_key"] == card_key:
                    a["removed"] = True

        _recent_actions.append({
            "dkey": dkey, "card_key": card_key, "app": app, "action": action,
            "summary": summary, "created_at": now, "removed": is_removal,
        })

        # Hand the intent string to the agentic router (src/server.ts -> route()):
        # a specialized Fetch.ai agent, the calendar tier, or a Browserbase web
        # agent that can accomplish essentially any web task. It runs in the
        # background — the optimistic card below shows immediately; the real
        # outcome streams back via _action_sink when the agent finishes.
        # Removals/undo aren't dispatched (no executor for reversing a live web
        # action yet) — they only strike the matching card in the HUD.
        if not is_removal:
            router_client.dispatch(summary, sink=_action_sink)
        return f"🧩 {summary}  ·  [{app}/{action}] ⟦{card_key}⟧"

    if name == "clarification_needed":
        q = (args.get("question") or "Which one did you mean?").strip()
        opts = args.get("options") or []
        if isinstance(opts, str):
            opts = [opts]
        opts = [str(o).strip() for o in opts if str(o).strip()]
        # Encoded so the server can render it as a question with options.
        return "❓CLARIFY|" + q + "|" + "||".join(opts)

    return f"[unknown tool: {name}]"
