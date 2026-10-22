import type { ErrorEnvelope, HealthResponse, ModelsResponse } from "./contracts/generated";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly envelope: ErrorEnvelope | null,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" }, signal });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const envelope =
      body && typeof body === "object" && "error" in body ? (body as ErrorEnvelope) : null;
    throw new ApiError(envelope?.error.message ?? `HTTP ${res.status}`, res.status, envelope);
  }
  return body as T;
}

export const getHealth = (signal?: AbortSignal) => getJson<HealthResponse>("/api/health", signal);
export const getModels = (signal?: AbortSignal) => getJson<ModelsResponse>("/api/models", signal);
