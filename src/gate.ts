import { AgentSearchResult } from "./agentverse_search";
import { llmJson } from "./llm";

const CONFIDENCE_THRESHOLD = 0.6;
// With community-noise filtered out (AGENTVERSE_FILTERS=-is:community), the
// remaining candidates are mostly real agents — judging only the top few
// misses good matches ranked lower by relevancy (e.g. a perfect-fit agent at
// position 7). Judge everything search returned; cost is parallel LLM calls,
// not sequential, so this stays fast.
const TOP_N = 10;

export interface Judgement {
  capable: boolean;
  confidence: number; // 0..1
  reason: string;
}

export interface GateOutcome {
  agent: AgentSearchResult;
  judgement: Judgement;
}

// The "brain": don't trust the top search hit. Ask whether the agent's own
// README actually covers the task, with a confidence we can threshold on.
async function judge(
  intent: string,
  agent: AgentSearchResult
): Promise<Judgement> {
  const system =
    `You decide whether a specialized agent can fulfill a user's task. ` +
    `Base the decision ONLY on the agent's described capabilities. ` +
    `Be strict: a vaguely related agent is NOT capable. ` +
    `Return ONLY JSON: {"capable": boolean, "confidence": number 0..1, "reason": string}`;
  const user = JSON.stringify({
    task: intent,
    agentName: agent.name,
    agentCapabilities: agent.readme?.slice(0, 2000) ?? "",
  });

  try {
    return await llmJson<Judgement>(system, user);
  } catch {
    // If the judge call fails, treat as not-capable so we fall through safely.
    return { capable: false, confidence: 0, reason: "judge call failed" };
  }
}

// Returns the best candidate that clears the bar, or null = "no good agent".
export async function pickBestAgent(
  intent: string,
  candidates: AgentSearchResult[],
  trace: string[]
): Promise<GateOutcome | null> {
  if (candidates.length === 0) return null;

  const judged = await Promise.all(
    candidates.slice(0, TOP_N).map(async (agent) => ({
      agent,
      judgement: await judge(intent, agent),
    }))
  );

  // Log every judgement — this is the demo-legible part.
  for (const j of judged) {
    trace.push(
      `gate: "${j.agent.name}" capable=${j.judgement.capable} ` +
        `conf=${j.judgement.confidence.toFixed(2)} — ${j.judgement.reason}`
    );
  }

  const best = judged
    .filter(
      (j) =>
        j.judgement.capable && j.judgement.confidence >= CONFIDENCE_THRESHOLD
    )
    .sort((a, b) => b.judgement.confidence - a.judgement.confidence)[0];

  if (!best) {
    trace.push(
      `gate: nothing cleared threshold ${CONFIDENCE_THRESHOLD} — falling through`
    );
    return null;
  }

  trace.push(`gate: selected "${best.agent.name}" (${best.agent.address})`);
  return best;
}
