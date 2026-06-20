import datetime
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"


def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def schedule_meeting(
    title: str,
    date: str,          # "YYYY-MM-DD"
    start_time: str,    # "HH:MM" 24-hour
    duration_minutes: int = 60,
    description: str = "",
    attendees: list[str] = [],
    timezone: str = "America/New_York",
) -> dict:
    service = get_calendar_service()

    start_dt = datetime.datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": timezone},
    }
    if attendees:
        event["attendees"] = [{"email": e} for e in attendees]

    created = service.events().insert(calendarId="primary", body=event, sendUpdates="all").execute()
    return created


def list_upcoming(max_results: int = 10):
    service = get_calendar_service()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    return events_result.get("items", [])


if __name__ == "__main__":
    # --- Schedule a meeting ---
    event = schedule_meeting(
        title="Shopping Review",
        date="2026-06-23",
        start_time="10:00",
        duration_minutes=30,
        description="Review best deals found by the scraper.",
        attendees=[],           # add emails here e.g. ["friend@gmail.com"]
        timezone="America/Los_Angeles",
    )
    print(f"Created: {event['summary']}")
    print(f"Link:    {event.get('htmlLink')}")

    # --- List upcoming events ---
    print("\nUpcoming events:")
    for e in list_upcoming():
        start = e["start"].get("dateTime", e["start"].get("date"))
        print(f"  {start}  {e['summary']}")
