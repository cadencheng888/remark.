// Discovery against the real Agentverse search API.
// POST https://agentverse.ai/v1/search/agents  (bearer token).
// Returns marketplace agents with a `readme` we can feed to the confidence gate.

const SEARCH_URL = "https://agentverse.ai/v1/search/agents";

// Shape of an agent object as returned by the search API (the fields we use).
export interface AgentSearchResult {
  address: string;
  name: string;
  readme: string; // self-described capabilities — the gate reads this
  status: string; // "active" when healthy
  total_interactions: number;
  recent_interactions: number;
  rating: number | null;
  type: string; // "hosted" | "local" | ...
  category: string;
  featured: boolean;
}

export interface SearchOptions {
  limit?: number;
  minRecentInteractions?: number; // cheap liveness floor before the LLM judge
  activeOnly?: boolean;
}

export async function searchAgents(
  searchText: string,
  trace: string[],
  opts: SearchOptions = {}
): Promise<AgentSearchResult[]> {
  const token = process.env.AGENTVERSE_API_KEY;
  if (!token) {
    trace.push("agentverse: no AGENTVERSE_API_KEY set — skipping discovery");
    return [];
  }

  const limit = opts.limit ?? 10;

  // Agentverse supports GitHub-style inline filters in search_text (is:active,
  // is:hosted, is:verified, is:fetch-ai). Injecting is:active up front asks the
  // API for live agents instead of us filtering out dead ones after the fact.
  // Configurable via AGENTVERSE_FILTERS (space-separated), defaults to "is:active".
  const inlineFilters = process.env.AGENTVERSE_FILTERS ?? "is:active";
  const searchWithFilters = inlineFilters
    ? `${searchText} ${inlineFilters}`.trim()
    : searchText;

  const body = {
    filters: { state: [], category: [], agent_type: [], protocol_digest: [] },
    sort: "relevancy",
    direction: "asc",
    search_text: searchWithFilters,
    offset: 0,
    limit,
  };
  trace.push(`agentverse: querying "${searchWithFilters}"`);

  let results: AgentSearchResult[] = [];
  try {
    const res = await fetch(SEARCH_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      trace.push(`agentverse: search HTTP ${res.status}`);
      return [];
    }
    const json = (await res.json()) as
      | { agents?: AgentSearchResult[] }
      | AgentSearchResult[];
    // API has returned either a bare array or an object wrapper across versions —
    // handle both so a shape change doesn't silently zero us out.
    results = Array.isArray(json) ? json : json.agents ?? [];
  } catch (e) {
    trace.push(`agentverse: search failed (${(e as Error).message})`);
    return [];
  }

  trace.push(`agentverse: search returned ${results.length} candidate(s)`);

  // Per-agent detail is useful when hunting for coverage, but noisy in a demo.
  // Gate it behind DEBUG_AGENTVERSE so the clean run stays clean.
  if (process.env.DEBUG_AGENTVERSE) {
    for (const a of results) {
      console.log(
        `    [debug] ${a.name} [${a.status}, recent=${a.recent_interactions}]`
      );
    }
  }

  // Cheap pre-filters: drop obviously-dead agents before spending LLM calls.
  const activeOnly = opts.activeOnly ?? true;
  const minRecent = opts.minRecentInteractions ?? 0;
  const filtered = results.filter((a) => {
    // Only DROP when the field exists and clearly disqualifies — if the real
    // response shape differs from the docs, we keep the agent rather than
    // silently zeroing out the whole list.
    if (activeOnly && a.status && a.status !== "active") return false;
    if (a.recent_interactions != null && a.recent_interactions < minRecent)
      return false;
    return true;
  });

  if (filtered.length !== results.length) {
    trace.push(`agentverse: ${filtered.length} left after liveness filter`);
  }
  return filtered;
}
