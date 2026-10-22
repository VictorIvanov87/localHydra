// Validates the shared SYNTHETIC fixtures in ../contracts/fixtures against the generated JSON Schemas
// (the frontend-side half of "contract fixture validation"), and checks the TS sequence tracker.
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { SequenceTracker } from "../src/contracts/sequence";

const root = resolve(__dirname, "../../contracts");
const read = (...p: string[]) => JSON.parse(readFileSync(join(root, ...p), "utf8"));

const manifest = read("schemas", "MANIFEST.json");
// pydantic emits an OpenAPI `discriminator` (with mapping) next to `oneOf`; Ajv cannot enforce
// mappings, so it is registered as an annotation. Each variant's `type` const still makes the
// oneOf unambiguous. strictTypes is off only because that keyword has no sibling `type`.
const ajv = new Ajv2020({ strict: true, strictTypes: false, allErrors: false });
ajv.addKeyword({ keyword: "discriminator" });
addFormats(ajv);
const validators = new Map<string, ReturnType<typeof ajv.compile>>();
for (const [name, meta] of Object.entries<{ file: string }>(manifest.schemas)) {
  validators.set(name, ajv.compile(read("schemas", meta.file)));
}

interface SchemaCase {
  name: string;
  schema: string;
  valid: boolean;
  schema_valid?: boolean;
  data: unknown;
}

const caseFiles = ["models", "metrics", "routing", "events", "api"];

describe("fixtures are synthetic and labeled", () => {
  for (const f of readdirSync(join(root, "fixtures")).filter((x) => x.endsWith(".json"))) {
    it(f, () => {
      const doc = read("fixtures", f);
      expect(doc.synthetic).toBe(true);
      expect(doc.label).toContain("SYNTHETIC");
    });
  }
});

describe("schema fixtures validate against generated JSON Schemas", () => {
  for (const file of caseFiles) {
    const doc = read("fixtures", `${file}.json`) as { cases: SchemaCase[] };
    for (const c of doc.cases) {
      it(`${file}: ${c.name}`, () => {
        const validate = validators.get(c.schema);
        expect(validate, `schema ${c.schema} registered`).toBeDefined();
        // JSON Schema cannot express cross-field invariants; those cases carry schema_valid=true
        // and are rejected by the Python reference instead.
        const expected = c.schema_valid ?? c.valid;
        const ok = validate!(c.data);
        expect(ok, JSON.stringify(validate!.errors)).toBe(expected);
      });
    }
  }
});

describe("every event in every stream fixture is schema-valid", () => {
  const streams = read("fixtures", "streams.json") as {
    cases: { name: string; events: unknown[] }[];
  };
  const validate = validators.get("Event")!;
  for (const c of streams.cases) {
    it(c.name, () => {
      for (const e of c.events) expect(validate(e), JSON.stringify(validate.errors)).toBe(true);
    });
  }
});

describe("SequenceTracker conformance (shared sse.json)", () => {
  const sse = read("fixtures", "sse.json") as {
    tracker: { name: string; sequences: number[]; expected: string[]; last_after?: number }[];
  };
  for (const c of sse.tracker) {
    it(c.name, () => {
      const t = new SequenceTracker();
      expect(c.sequences.map((s) => t.accept(s))).toEqual(c.expected);
      if (c.last_after !== undefined) expect(t.last).toBe(c.last_after);
    });
  }
});

describe("endpoint registry", () => {
  it("references only registered schemas", () => {
    const endpoints = read("schemas", "endpoints.json") as {
      request_schema: string | null;
      response_schema: string | null;
    }[];
    for (const e of endpoints) {
      for (const n of [e.request_schema, e.response_schema]) {
        if (n) expect(validators.has(n)).toBe(true);
      }
    }
  });
});
