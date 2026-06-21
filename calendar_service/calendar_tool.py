"""
Google Calendar + Google Tasks integration (the "external programs").
Auth uses Google's OAuth desktop flow:
  1. Create a Google Cloud project, enable the Google Calendar API and Tasks API.
  2. Create an OAuth client ID of type "Desktop app".
  3. Download it as credentials.json into this folder.
First run opens a browser to grant access; the token is cached in token.json.
"""
import datetime
import os
import zoneinfo
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
]
TZ_NAME = os.environ.get("LOCAL_TIMEZONE", "America/Los_Angeles")
def _credentials() -> Credentials:
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return creds
def create_event(title, start_iso, duration_minutes=60, location=None, notes=None) -> dict:
    service = build("calendar", "v3", credentials=_credentials())
    start = datetime.datetime.fromisoformat(start_iso)
    end = start + datetime.timedelta(minutes=duration_minutes)
    body = {
        "summary": title,
        "start": {"dateTime": start.isoformat(), "timeZone": TZ_NAME},
        "end": {"dateTime": end.isoformat(), "timeZone": TZ_NAME},
    }
    if location:
        body["location"] = location
    if notes:
        body["description"] = notes
    event = service.events().insert(calendarId="primary", body=body).execute()
    return {
        "message": f"📅 Calendar event created: '{title}' at {start_iso} -> {event.get('htmlLink')}",
        "event_id": event.get("id"),
        "html_link": event.get("htmlLink"),
    }
def update_event(event_id, title=None, start_iso=None, duration_minutes=None, location=None, notes=None) -> dict:
    service = build("calendar", "v3", credentials=_credentials())
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    if title:
        event["summary"] = title
    if start_iso:
        start = datetime.datetime.fromisoformat(start_iso)
        # Preserve existing duration unless a new one is given.
        if duration_minutes is None:
            old_start = datetime.datetime.fromisoformat(event["start"]["dateTime"])
            old_end = datetime.datetime.fromisoformat(event["end"]["dateTime"])
            duration_minutes = int((old_end - old_start).total_seconds() / 60)
        end = start + datetime.timedelta(minutes=duration_minutes)
        event["start"] = {"dateTime": start.isoformat(), "timeZone": TZ_NAME}
        event["end"] = {"dateTime": end.isoformat(), "timeZone": TZ_NAME}
    if location:
        event["location"] = location
    if notes:
        event["description"] = notes
    updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return {
        "message": f"✏️  Calendar event updated: '{updated.get('summary')}' -> {updated.get('htmlLink')}",
        "event_id": updated.get("id"),
        "html_link": updated.get("htmlLink"),
    }
def list_events(time_min_iso=None, time_max_iso=None, max_results=20) -> list:
    service = build("calendar", "v3", credentials=_credentials())
    time_min = (
        datetime.datetime.fromisoformat(time_min_iso).astimezone(zoneinfo.ZoneInfo("UTC"))
        if time_min_iso
        else datetime.datetime.now(zoneinfo.ZoneInfo("UTC"))
    )
    params = {
        "calendarId": "primary",
        "timeMin": time_min.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if time_max_iso:
        time_max = datetime.datetime.fromisoformat(time_max_iso).astimezone(zoneinfo.ZoneInfo("UTC"))
        params["timeMax"] = time_max.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    events = service.events().list(**params).execute().get("items", [])
    return [
        {
            "event_id": e.get("id"),
            "title": e.get("summary", "(no title)"),
            "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
            "html_link": e.get("htmlLink"),
        }
        for e in events
    ]
def delete_event(event_id: str) -> str:
    service = build("calendar", "v3", credentials=_credentials())
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return f"🗑️  Calendar event deleted (id {event_id})"
def create_task(title, due_iso=None) -> str:
    service = build("tasks", "v1", credentials=_credentials())
    body = {"title": title}
    if due_iso:
        due = datetime.datetime.fromisoformat(due_iso).astimezone(zoneinfo.ZoneInfo("UTC"))
        body["due"] = due.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    task = service.tasks().insert(tasklist="@default", body=body).execute()
    return f"☑️  Task created: '{title}' (id {task.get('id')})"