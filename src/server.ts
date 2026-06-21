// HTTP wrapper around route() so the Python audio pipeline can hand off an
// intent string and get back the router's structured result.
//
//   npm run serve            (tsx --env-file=.env src/server.ts)
//   POST /route  { "intent": "add AirPods to my cart" }  -> RouteResult JSON
//   GET  /health
//
// This is the missing bridge between the perception half (Python: mic ->
// Deepgram -> Claude -> intent string) and the execution half (this TS router:
// Agentverse / Calendar / Browserbase). Any intent is accepted — calendar is
// just one tier inside route().
import { createServer, type ServerResponse } from "node:http";
import { route } from "./router";

const PORT = Number(process.env.ROUTER_PORT) || 8788;

function send(res: ServerResponse, code: number, obj: unknown) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}

const server = createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    return send(res, 200, { ok: true });
  }
  if (req.method !== "POST" || req.url !== "/route") {
    return send(res, 404, { error: "use POST /route or GET /health" });
  }

  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", async () => {
    let intent = "";
    try {
      intent = String(JSON.parse(body).intent ?? "").trim();
    } catch {
      return send(res, 400, { error: "body must be JSON: { intent: string }" });
    }
    if (!intent) return send(res, 400, { error: "missing 'intent'" });

    console.log(`\n→ route("${intent}")`);
    // Stream newline-delimited JSON: one {type:"trace"} per reasoning line as it
    // happens, then a final {type:"result"}. Lets the HUD show the agent think
    // live during a 30-90s Browserbase run instead of a silent wait.
    res.writeHead(200, {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache",
    });
    const write = (obj: unknown) => res.write(JSON.stringify(obj) + "\n");
    try {
      const result = await route(intent, (line) => write({ type: "trace", line }));
      console.log(`← ${result.source}/${result.status}`);
      write({ type: "result", result });
    } catch (e) {
      console.error("route failed:", e);
      write({
        type: "result",
        result: {
          source: "none",
          status: "failed",
          payload: { error: (e as Error).message },
          trace: [],
        },
      });
    }
    res.end();
  });
});

server.listen(PORT, () => {
  console.log(`intent router listening → http://localhost:${PORT}/route`);
});
