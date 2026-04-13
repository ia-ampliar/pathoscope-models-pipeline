export async function fetchSchema() {
  const r = await fetch("/api/schema");
  if (!r.ok) throw new Error("Schema indisponível");
  return r.json() as Promise<import("./types").ApiSchemaPayload>;
}

export function wsTrainUrl(jobId: string) {
  if (import.meta.env.DEV) {
    return `ws://127.0.0.1:8000/api/train/jobs/${jobId}/stream`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/train/jobs/${jobId}/stream`;
}
