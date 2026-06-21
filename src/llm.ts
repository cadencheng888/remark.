// Minimal OpenAI-compatible chat client. Works with any provider that exposes
// /v1/chat/completions — set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env.
// Examples:
//   OpenAI:    https://api.openai.com/v1            gpt-4o-mini
//   ASI:One:   https://api.asi1.ai/v1               asi1-mini      (Fetch.ai's own LLM)
//   Anthropic: via an OpenAI-compatible gateway of your choice
//
// We deliberately don't hardcode a provider so the gate works with whatever
// key you already have on hand.

const BASE_URL = process.env.LLM_BASE_URL ?? "https://api.openai.com/v1";
const MODEL = process.env.LLM_MODEL ?? "gpt-4o-mini";

export async function llmJson<T>(system: string, user: string): Promise<T> {
  const key = process.env.LLM_API_KEY;
  if (!key)
    throw new Error("LLM_API_KEY not set — needed for the confidence gate");

  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0,
    }),
  });

  if (!res.ok) {
    throw new Error(`LLM HTTP ${res.status}: ${await res.text()}`);
  }

  const data = (await res.json()) as {
    choices: { message: { content: string } }[];
  };
  const text = data.choices?.[0]?.message?.content ?? "";
  const clean = text.replace(/```json|```/g, "").trim();
  return JSON.parse(clean) as T;
}
