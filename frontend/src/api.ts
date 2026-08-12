const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface SourceStem {
  stem: string;
  start_seconds: number;
  audio_url: string | null;
}

export interface QueryResponse {
  answer: string;
  source_stems: SourceStem[];
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export async function queryRAG(
  question: string,
  history: HistoryMessage[]
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listTranscripts(): Promise<string[]> {
  const res = await fetch(`${BASE}/transcripts`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.stems;
}

export async function getAudioUrl(stem: string): Promise<string> {
  const res = await fetch(`${BASE}/transcripts/${stem}/audio`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.url;
}

export interface PipelineEvent {
  type: "log" | "done" | "error";
  data: Record<string, string>;
}

export function runPipeline(
  formData: FormData,
  onEvent: (event: PipelineEvent) => void
): () => void {
  const controller = new AbortController();

  fetch(`${BASE}/pipeline/run`, {
    method: "POST",
    body: formData,
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      onEvent({ type: "error", data: { message: await res.text() } });
      return;
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const lines = part.trim().split("\n");
        let eventType = "log";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7);
          if (line.startsWith("data: ")) dataStr = line.slice(6);
        }
        if (dataStr) {
          try {
            onEvent({ type: eventType as PipelineEvent["type"], data: JSON.parse(dataStr) });
          } catch {}
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") {
      onEvent({ type: "error", data: { message: String(err) } });
    }
  });

  return () => controller.abort();
}

export function resumePipeline(
  params: Record<string, string>,
  onEvent: (event: PipelineEvent) => void
): () => void {
  const controller = new AbortController();

  fetch(`${BASE}/pipeline/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      onEvent({ type: "error", data: { message: await res.text() } });
      return;
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const lines = part.trim().split("\n");
        let eventType = "log";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7);
          if (line.startsWith("data: ")) dataStr = line.slice(6);
        }
        if (dataStr) {
          try {
            onEvent({ type: eventType as PipelineEvent["type"], data: JSON.parse(dataStr) });
          } catch {}
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") {
      onEvent({ type: "error", data: { message: String(err) } });
    }
  });

  return () => controller.abort();
}
