import { route } from "./router";
import { RouteResult } from "./types";

const DIV = "─".repeat(64);

function header(text: string) {
  console.log("\n" + DIV);
  console.log("  " + text);
  console.log(DIV);
}

function labeledSource(source: RouteResult["source"]): string {
  switch (source) {
    case "agentverse":
      return "Fetch.ai Agentverse  (specialized agent)";
    case "calendar":
      return "Calendar Action      (Google Calendar API)";
    case "browserbase":
      return "Browserbase          (autonomous web agent)";
    default:
      return source;
  }
}

function renderTrace(trace: string[]) {
  header("ROUTING");
  for (const line of trace) {
    const idx = line.indexOf(":");
    if (idx === -1) {
      console.log("  " + line);
      continue;
    }
    const tier = line.slice(0, idx).trim();
    const msg = line.slice(idx + 1).trim();

    // Gate lines carry a verdict — flag it visibly so the one match (if any)
    // doesn't read identically to the rejections around it.
    let marker = "";
    if (tier === "gate") {
      if (msg.startsWith("selected ")) {
        marker = "  [MATCH]   ";
      } else if (msg.includes("capable=true")) {
        marker = "  [PASS]    ";
      } else if (msg.includes("capable=false")) {
        marker = "  [reject]  ";
      }
    }

    console.log("  " + tier.padEnd(12) + marker + msg);
  }
}

function renderResult(result: RouteResult) {
  header("RESULT");
  console.log("  Handled by   " + labeledSource(result.source));
  console.log("  Status       " + result.status.toUpperCase());

  const p = result.payload as Record<string, unknown> | null;
  const selected = p?.agentverseSelected as
    | { name: string; address: string; confidence: number }
    | undefined;
  if (selected) {
    console.log("");
    console.log("  Agentverse selected:  " + selected.name);
    console.log("  Confidence:           " + selected.confidence.toFixed(2));
    console.log("  Address:              " + selected.address);
    console.log(
      "  (invocation not yet wired — task completed by " +
        labeledSource(result.source).trim() +
        ")"
    );
  }
  console.log("");

  if (!p) {
    console.log("  (no payload)");
    return;
  }

  const preferred = [
    "outcome",
    "details",
    "actionTaken",
    "stoppedBecause",
    "message",
  ];
  for (const key of preferred) {
    if (p[key] != null && p[key] !== "") {
      console.log("  " + key.padEnd(15) + String(p[key]));
    }
  }
  for (const [key, val] of Object.entries(p)) {
    if (preferred.includes(key) || key === "agentverseSelected") continue;
    if (val == null || val === "") continue;
    const rendered =
      typeof val === "object" ? JSON.stringify(val) : String(val);
    console.log("  " + key.padEnd(15) + rendered);
  }
}

async function main() {
  const intent = process.argv.slice(2).join(" ").trim();
  if (!intent) {
    console.error('Usage: npm run dev -- "the user wants to do X"');
    process.exit(1);
  }

  header("INTENT");
  console.log("  " + intent);

  const result = await route(intent);

  renderTrace(result.trace);
  renderResult(result);
  console.log("");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
