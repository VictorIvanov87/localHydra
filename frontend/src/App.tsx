import { useEffect, useState } from "react";
import { getHealth, getModels } from "./api";
import type { HealthResponse, ModelsResponse } from "./contracts/generated";

type Loaded =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; health: HealthResponse; models: ModelsResponse };

export function App() {
  const [data, setData] = useState<Loaded>({ state: "loading" });

  useEffect(() => {
    const ctl = new AbortController();
    Promise.all([getHealth(ctl.signal), getModels(ctl.signal)])
      .then(([health, models]) => setData({ state: "ready", health, models }))
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return;
        setData({ state: "error", message: e instanceof Error ? e.message : String(e) });
      });
    return () => ctl.abort();
  }, []);

  return (
    <main>
      <h1>localHydra</h1>
      <p className="tagline">Observable, local-only model routing. Dashboard shell (LH-001).</p>

      {data.state === "loading" && <p role="status">Connecting to the local backend…</p>}
      {data.state === "error" && (
        <p role="alert" className="bad">
          Backend unreachable: {data.message}
        </p>
      )}
      {data.state === "ready" && (
        <>
          <section aria-labelledby="health-h">
            <h2 id="health-h">Backend</h2>
            <dl>
              <dt>Status</dt>
              <dd data-testid="health-status">{data.health.status}</dd>
              <dt>Bind host</dt>
              <dd>{data.health.bind_host}</dd>
              <dt>Ollama target</dt>
              <dd>{data.health.ollama_target}</dd>
              <dt>Contract</dt>
              <dd>
                v{data.health.contract_version} (events schema {data.health.event_schema_version})
              </dd>
            </dl>
          </section>
          <section aria-labelledby="models-h">
            <h2 id="models-h">Models</h2>
            {!data.models.configured ? (
              <p>No model config supplied. Start the backend with --config (see docs/development.md).</p>
            ) : (
              <ul>
                {data.models.models.map((m) => (
                  <li key={m.profile.id}>
                    <code>{m.profile.tag}</code> — {m.availability}
                    {m.availability_reason ? ` (${m.availability_reason})` : ""}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </main>
  );
}
