import { RawIntent, RouteResult } from "./types";
import { llmJson } from "./llm";

const CAL_BASE = process.env.CALENDAR_SERVICE_URL ?? "http://localhost:8787";

// What we ask the LLM to extract from a free-text intent before calling the
// calendar service. Keeping this narrow (one of three actions) makes the
// extraction reliable — we're not asking it to handle arbitrary calendar ops.
interface CalendarPlan {
  action: "create" | "update" | "delete" | "none";
  title?: string;
  start_iso?: string; // ISO 8601, local time, no offset (service applies TZ)
  duration_minutes?: number;
  location?: string;
  notes?: string;
  // For update/delete: a description used to find the target event via list_events.
  target_description?: string;
}

async function planCalendarAction(intent: RawIntent): Promise<CalendarPlan> {
  const now = new Date().toISOString();
  const system =
    `You turn a user's calendar request into a structured action. ` +
    `Today's date/time is ${now}. ` +
    `Return ONLY JSON matching: { "action": "create"|"update"|"delete"|"none", ` +
    `"title"?: string, "start_iso"?: string (no timezone offset), ` +
    `"duration_minutes"?: number, "location"?: string, "notes"?: string, ` +
    `"target_description"?: string }. ` +
    `Use "none" if this isn't actually a calendar create/update/delete request. ` +
    `For update/delete, "target_description" should describe which existing event is meant.`;

  return llmJson<CalendarPlan>(system, intent);
}

async function findEventId(
  description: string,
  trace: string[]
): Promise<string | null> {
  const res = await fetch(`${CAL_BASE}/events`);
  if (!res.ok) return null;
  const events = (await res.json()) as { event_id: string; title: string }[];
  if (events.length === 0) return null;

  // Cheap heuristic match first; LLM disambiguation could replace this later.
  const lower = description.toLowerCase();
  const hit = events.find(
    (e) =>
      e.title.toLowerCase().includes(lower) ||
      lower.includes(e.title.toLowerCase())
  );
  if (hit)
    trace.push(
      `calendar: matched "${description}" -> "${hit.title}" (${hit.event_id})`
    );
  return hit?.event_id ?? null;
}

// Returns null if this doesn't look like a calendar action (so the router can
// try the next tier), or a RouteResult on success/failure of a real attempt.
export async function tryCalendarAction(
  intent: RawIntent,
  trace: string[]
): Promise<RouteResult | null> {
  let plan: CalendarPlan;
  try {
    plan = await planCalendarAction(intent);
  } catch (e) {
    trace.push(`calendar: planning failed (${(e as Error).message})`);
    return null;
  }

  if (plan.action === "none") {
    trace.push("calendar: intent is not a calendar action");
    return null;
  }
  trace.push(`calendar: planned action="${plan.action}"`);

  try {
    if (plan.action === "create") {
      if (!plan.title || !plan.start_iso) {
        trace.push(
          "calendar: missing title/start_iso for create — falling through"
        );
        return null;
      }
      const res = await fetch(`${CAL_BASE}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: plan.title,
          start_iso: plan.start_iso,
          duration_minutes: plan.duration_minutes ?? 60,
          location: plan.location,
          notes: plan.notes,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(body));
      trace.push("calendar: event created");
      return { source: "calendar", status: "success", payload: body, trace };
    }

    if (plan.action === "delete" || plan.action === "update") {
      const targetDesc = plan.target_description ?? plan.title ?? intent;
      const eventId = await findEventId(targetDesc, trace);
      if (!eventId) {
        trace.push(
          `calendar: no matching event found for "${targetDesc}" — falling through`
        );
        return null;
      }

      if (plan.action === "delete") {
        const res = await fetch(`${CAL_BASE}/events/${eventId}`, {
          method: "DELETE",
        });
        const body = await res.json();
        if (!res.ok) throw new Error(JSON.stringify(body));
        trace.push("calendar: event deleted");
        return { source: "calendar", status: "success", payload: body, trace };
      }

      // update
      const res = await fetch(`${CAL_BASE}/events/${eventId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: plan.title,
          start_iso: plan.start_iso,
          duration_minutes: plan.duration_minutes,
          location: plan.location,
          notes: plan.notes,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(body));
      trace.push("calendar: event updated");
      return { source: "calendar", status: "success", payload: body, trace };
    }

    return null;
  } catch (e) {
    trace.push(`calendar: action failed (${(e as Error).message})`);
    return { source: "calendar", status: "failed", payload: null, trace };
  }
}
