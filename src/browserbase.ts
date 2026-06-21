import { Stagehand } from "@browserbasehq/stagehand";
import Browserbase from "@browserbasehq/sdk";
import { exec } from "node:child_process";
import { z } from "zod";
import { RawIntent, RouteResult } from "./types";

// disableAPI: true means Stagehand calls the model provider directly instead
// of going through Browserbase's hosted Model Gateway — so it needs a real
// provider API key here, separate from BROWSERBASE_API_KEY.
const MODEL_NAME = process.env.STAGEHAND_MODEL ?? "anthropic/claude-sonnet-4-6";
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;

// Cross-platform "open URL in default browser" without adding a dependency.
function openInBrowser(url: string) {
  const cmd =
    process.platform === "darwin"
      ? `open "${url}"`
      : process.platform === "win32"
      ? `start "" "${url}"`
      : `xdg-open "${url}"`;
  exec(cmd, (err) => {
    if (err) console.error("Could not auto-open live view:", err.message);
  });
}

// Fetches the Browserbase live-view URL for a session and opens it immediately,
// so a judge/audience can watch the agent drive the browser in real time.
// This is best-effort: a failure here should never abort the actual task.
async function openLiveView(sessionId: string | undefined, trace: string[]) {
  if (!sessionId) {
    trace.push("browserbase: no session id yet — skipping live view");
    return;
  }
  try {
    const bb = new Browserbase({ apiKey: process.env.BROWSERBASE_API_KEY });
    const liveUrls = await bb.sessions.debug(sessionId);
    trace.push(`browserbase: live view -> ${liveUrls.debuggerFullscreenUrl}`);
    openInBrowser(liveUrls.debuggerFullscreenUrl);
  } catch (e) {
    trace.push(
      `browserbase: couldn't open live view (${(e as Error).message})`
    );
  }
}

// Generalized fallback: hosted browser + agentic loop that "figures out" the task.
// Stagehand v3: pass an `output` schema INTO agent.execute() so structured data
// comes back in the same call — no separate extract() afterward (which races
// against the now-closed session and throws -32001 "session not found").
export async function runBrowserbase(
  intent: RawIntent,
  trace: string[]
): Promise<RouteResult> {
  trace.push(`browserbase: handling "${intent}"`);

  if (!ANTHROPIC_KEY) {
    trace.push(
      "browserbase: ANTHROPIC_API_KEY not set — required in direct/disableAPI mode"
    );
    return { source: "browserbase", status: "failed", payload: null, trace };
  }

  const stagehand = new Stagehand({
    env: "BROWSERBASE",
    apiKey: process.env.BROWSERBASE_API_KEY,
    projectId: process.env.BROWSERBASE_PROJECT_ID,
    // Object form (not the bare string) is required so we can attach apiKey —
    // see ModelConfiguration: ClientOptions & { modelName } is direct-mode only.
    model: {
      modelName: MODEL_NAME,
      apiKey: ANTHROPIC_KEY,
    },
    // Required for agent.execute({ output: zodSchema }) — structured agent
    // output is an experimental feature and only works outside the hosted
    // Stagehand API path. See: Stagehand error "Agent output schema is
    // experimental... set experimental: true and disableAPI: true".
    experimental: true,
    disableAPI: true,
    verbose: 1,
  });

  try {
    await stagehand.init();

    // Open the live view right away so the audience watches from the first action.
    await openLiveView(stagehand.browserbaseSessionID, trace);

    // Sane starting surface: an auth-free search engine.
    const page = stagehand.context.pages()[0];
    await page.goto("https://duckduckgo.com");

    // agent() runs the multi-step loop AND returns structured output in one shot.
    const agent = stagehand.agent();
    const run = await agent.execute({
      instruction:
        `Accomplish this task on the web: ${intent}.\n` +
        `Navigate as needed and take concrete actions (search, click, fill fields, ` +
        `add items to a cart, start a booking/reservation form).\n` +
        `SAFETY RULES — follow strictly:\n` +
        `- If the task is informational, find and report the answer.\n` +
        `- For shopping: add the item to the cart, then STOP at the cart page. ` +
        `Do NOT proceed to checkout, do NOT enter payment details.\n` +
        `- For reservations/bookings: fill the form up to the final confirmation ` +
        `step, then STOP. Do NOT submit a final booking or enter payment.\n` +
        `- NEVER log in, create an account, or enter passwords or credit cards.\n` +
        `- If a step requires login/payment to continue, stop there and report ` +
        `exactly how far you got.`,
      output: z.object({
        outcome: z.string().describe("what was accomplished or found"),
        details: z
          .string()
          .describe("key specifics: values, links, confirmation"),
        actionTaken: z
          .string()
          .describe(
            "the concrete action performed, e.g. 'added MacBook Air to cart', or 'none' if read-only"
          ),
        stoppedBecause: z
          .string()
          .describe(
            "why the agent stopped if it didn't fully complete, e.g. 'reached login wall', or 'completed' otherwise"
          ),
        success: z
          .boolean()
          .describe("whether the task was actually completed"),
      }),
    });
    trace.push(`browserbase: agent finished (completed=${run.completed})`);

    // Prefer the structured output; fall back to the agent's message if absent.
    const out = (run.output ?? {}) as {
      outcome?: string;
      details?: string;
      actionTaken?: string;
      stoppedBecause?: string;
      success?: boolean;
    };
    const succeeded = out.success ?? run.success;

    return {
      source: "browserbase",
      status: succeeded ? "success" : "partial",
      payload: {
        outcome: out.outcome ?? run.message,
        details: out.details ?? "",
        actionTaken: out.actionTaken ?? "none",
        stoppedBecause: out.stoppedBecause ?? "",
        success: succeeded,
        actions: run.actions?.length ?? 0,
      },
      trace,
    };
  } catch (e) {
    trace.push(`browserbase: failed (${(e as Error).message})`);
    return { source: "browserbase", status: "failed", payload: null, trace };
  } finally {
    await stagehand.close();
  }
}
