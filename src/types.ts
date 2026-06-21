// The intent interface is a single string for now.
// When the glasses side sends structured context, widen this to an object.
export type RawIntent = string;

export type ResultSource = "agentverse" | "calendar" | "browserbase" | "none";
export type ResultStatus = "success" | "partial" | "failed";

export interface RouteResult {
  source: ResultSource;
  status: ResultStatus;
  payload: unknown; // whatever the executor returned (structured extract, text, etc.)
  trace: string[]; // human-readable log of what the router did — this is the demo narration
}
