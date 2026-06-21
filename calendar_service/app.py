"""
Minimal local HTTP wrapper around calendar_tool.py so the TypeScript router
can call it with plain fetch — no Python/Node interop needed.

Run:  uvicorn calendar_service.app:app --port 8787
Then: POST http://localhost:8787/events            (create)
      GET  http://localhost:8787/events             (list)
      PATCH http://localhost:8787/events/{event_id} (update)
      DELETE http://localhost:8787/events/{event_id}(delete)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import calendar_tool as cal

app = FastAPI(title="calendar-action-service")


class CreateEventBody(BaseModel):
    title: str
    start_iso: str
    duration_minutes: int = 60
    location: Optional[str] = None
    notes: Optional[str] = None


class UpdateEventBody(BaseModel):
    title: Optional[str] = None
    start_iso: Optional[str] = None
    duration_minutes: Optional[int] = None
    location: Optional[str] = None
    notes: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/events")
def create_event(body: CreateEventBody):
    try:
        return cal.create_event(
            title=body.title,
            start_iso=body.start_iso,
            duration_minutes=body.duration_minutes,
            location=body.location,
            notes=body.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events")
def list_events(time_min_iso: Optional[str] = None, time_max_iso: Optional[str] = None):
    try:
        return cal.list_events(time_min_iso=time_min_iso, time_max_iso=time_max_iso)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/events/{event_id}")
def update_event(event_id: str, body: UpdateEventBody):
    try:
        return cal.update_event(
            event_id=event_id,
            title=body.title,
            start_iso=body.start_iso,
            duration_minutes=body.duration_minutes,
            location=body.location,
            notes=body.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    try:
        message = cal.delete_event(event_id)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))