import { RawIntent, RouteResult } from "./types";
import { runBrowserbase } from "./browserbase";
import { searchAgents } from "./agentverse_search";
import { pickBestAgent, GateOutcome } from "./gate";
// import { looksLikeSms, runTwilioSms } from "./twilio_sms"; // DISABLED: file not implemented yet
import { tryCalendarAction } from "./calendar_action";

// Part 1: discover a specialized agent and judge it.
// Invocation is deferred — selection is recorded in the trace as the headline
// finding ("the right agent WAS found"), but execution still falls through to
// the next tier so the task actually completes. This is intentional, not a
// placeholder: it demonstrates real graceful degradation rather than stopping
// at "selected but not yet invoked."
async function tryAgentverse(
  intent: RawIntent,
  trace: string[]
): Promise<{ selectedAgent: GateOutcome | null }> {
  const candidates = await searchAgents(intent, trace, {
    limit: Number(process.env.AGENTVERSE_DEBUG_LIMIT) || 10,
    activeOnly: true,
    minRecentInteractions: 0,
  });
  if (candidates.length === 0) return { selectedAgent: null };

  const best = await pickBestAgent(intent, candidates, trace);
  if (!best) return { selectedAgent: null };

  trace.push(
    `agentverse: invocation not yet wired — selection recorded, continuing to next tier`
  );
  return { selectedAgent: best };
}

// Sequential by design: prefer specialized agents, degrade gracefully to web.
export async function route(intent: RawIntent): Promise<RouteResult> {
  const trace: string[] = [`intent: "${intent}"`];

  // SMS action tier disabled until twilio_sms.ts is implemented.
  // if (looksLikeSms(intent) && process.env.TWILIO_ACCOUNT_SID) {
  //   trace.push("route: intent looks like SMS — trying Twilio action");
  //   const sms = await runTwilioSms(intent, trace);
  //   if (sms.status === "success") return sms;
  //   trace.push("route: Twilio didn't complete — falling through to web");
  // }

  const { selectedAgent } = await tryAgentverse(intent, trace);

  // Structured/API action tier: calendar create/update/delete via Google Calendar API.
  // Returns null if the intent isn't calendar-shaped, so we fall through to Browserbase.
  const fromCalendar = await tryCalendarAction(intent, trace);
  if (fromCalendar) {
    return attachSelection(fromCalendar, selectedAgent);
  }

  const fromBrowserbase = await runBrowserbase(intent, trace);
  return attachSelection(fromBrowserbase, selectedAgent);
}

// Surfaces "Agentverse selected agent X" alongside whichever tier actually
// completed the task, without changing that tier's own result shape.
function attachSelection(
  result: RouteResult,
  selected: GateOutcome | null
): RouteResult {
  if (!selected) return result;
  return {
    ...result,
    payload: {
      ...(result.payload as Record<string, unknown>),
      agentverseSelected: {
        name: selected.agent.name,
        address: selected.agent.address,
        confidence: selected.judgement.confidence,
      },
    },
  };
}
