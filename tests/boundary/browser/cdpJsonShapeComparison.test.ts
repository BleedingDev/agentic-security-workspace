import { describe, expect, it } from "vitest";

import { compareWebCaptureEvidence } from "../../../src/application/BrowserObservationService.js";
import { CdpBrowserProvider } from "../../../src/browser/CdpBrowserProvider.js";
import { CdpCaptureEvents } from "../../../src/browser/CdpCaptureEvents.js";
import { browserCaptureComparisonInputSchema } from "../../../src/domain/browserCaptureComparison.js";
import {
  inspectWebPageInputSchema,
  webPageInspectionSchema,
  type WebPageInspection,
} from "../../../src/domain/browserObservation.js";
import { parseEvidence } from "../../../src/domain/evidence.js";
import { webCaptureDiffSchema } from "../../../src/domain/webCaptureDiff.js";

const ORIGIN = "https://example.test";
const CAPTURE_TIME = "2026-01-01T00:00:00.000Z";
type BodySource = "request" | "response" | "base64 response";

const equivalentBodies = [
  {
    name: "root keys",
    before: '{"é":true,"e\u0301":null}',
    after: '{"e\u0301":null,"é":true}',
  },
  {
    name: "nested null and array values",
    before: '{"outer":{"é":null,"e\u0301":[1,null]}}',
    after: '{"outer":{"e\u0301":[1,null],"é":null}}',
  },
  {
    name: "object arrays",
    before: '[{"é":1,"e\u0301":null},{"é":2,"e\u0301":[]}]',
    after: '[{"e\u0301":null,"é":1},{"e\u0301":[],"é":2}]',
  },
  {
    name: "escaped pointer names",
    before: '{"é/~":1,"e\u0301/~":true}',
    after: '{"e\u0301/~":true,"é/~":1}',
  },
];

describe.each<BodySource>(["request", "response", "base64 response"])(
  "CDP %s JSON shape comparison",
  (source) => {
    it.each(equivalentBodies)(
      "does not report a network modification for reordered $name",
      async ({ before: beforeBody, after: afterBody }) => {
        const [before, after] = comparableCaptures(
          source,
          beforeBody,
          afterBody,
        );
        const result = await compareCaptures(before, after);

        expect(result.dimensions.network).toEqual({
          status: "unknown",
          total_changes: 0,
          changes: [],
          reason: expect.any(String),
        });
        expect(result.overall_status).toBe("unknown");
        expect(before.network).toEqual(after.network);
      },
    );

    it.each([
      { name: "a type change", after: '{"é":null,"e\u0301":null}' },
      { name: "a removed key", after: '{"e\u0301":null}' },
      {
        name: "swapped key/type associations",
        after: '{"é":null,"e\u0301":true}',
      },
    ])("still reports $name", async ({ after: changedBody }) => {
      const beforeBody = '{"é":true,"e\u0301":null}';
      // JSON whitespace equalizes wire length even for the removed-key control.
      const afterBody =
        changedBody +
        " ".repeat(
          Buffer.byteLength(beforeBody) - Buffer.byteLength(changedBody),
        );
      const [before, after] = comparableCaptures(source, beforeBody, afterBody);
      const result = await compareCaptures(before, after);

      expect(result.overall_status).toBe("changed");
      expect(result.dimensions.network).toEqual({
        status: "changed",
        total_changes: 1,
        changes: [
          { identity: expect.stringMatching(/^net_/u), change: "modified" },
        ],
        reason: null,
      });
    });
  },
);

const comparableCaptures = (
  source: BodySource,
  beforeBody: string,
  afterBody: string,
): readonly [WebPageInspection, WebPageInspection] => {
  expect(Buffer.byteLength(beforeBody)).toBe(Buffer.byteLength(afterBody));
  const before = captureBody(source, beforeBody);
  const after = captureBody(source, afterBody);
  const withoutShapes = (capture: WebPageInspection) => ({
    ...capture,
    network: {
      ...capture.network,
      requests: capture.network.requests.map(
        ({ body_shapes: _shapes, ...request }) => request,
      ),
    },
  });
  // Prove no status, length, metadata, or other observed field confounds the diff.
  expect(withoutShapes(before)).toEqual(withoutShapes(after));
  for (const capture of [before, after]) {
    expect(capture.network.requests).toHaveLength(1);
    expect(capture.network.prior_activity_available).toBe(false);
    expect(capture.completeness.attach_limited_sections).toContain(
      "network_requests",
    );
    const request = capture.network.requests[0];
    expect(request).not.toHaveProperty("postData");
    expect(request).not.toHaveProperty("body");
    expect(request).not.toHaveProperty("body_hash");
    expect(request?.body_shapes.status).toBe("included");
    expect(
      request?.body_shapes[source === "request" ? "request" : "response"],
    ).not.toBeNull();
  }
  return [before, after];
};

const compareCaptures = async (
  before: WebPageInspection,
  after: WebPageInspection,
) => {
  const result = await compareWebCaptureEvidence(
    new CdpBrowserProvider(),
    browserCaptureComparisonInputSchema.parse({
      before: { inspection: before },
      after: { inspection: after },
    }),
  );
  if (!result.ok) throw result.error;
  const evidence = parseEvidence(JSON.parse(JSON.stringify(result.value)));
  expect(evidence.operation).toBe("compare_web_captures");
  return webCaptureDiffSchema.parse(evidence.normalized_result);
};

const captureBody = (source: BodySource, body: string): WebPageInspection => {
  const events = new CdpCaptureEvents(
    inspectWebPageInputSchema.parse({
      cdp_endpoint: "http://127.0.0.1:9222",
      allowed_origins: [ORIGIN],
      target_id: "page-1",
      observation_ms: 0,
      include_json_body_shapes: true,
    }),
    new Set([ORIGIN]),
  );
  events.beginAuthorizedFrame("main");
  events.ingest({
    method: "Network.requestWillBeSent",
    params: {
      requestId: "request-1",
      type: "Fetch",
      request: {
        url: `${ORIGIN}/api`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        postData: source === "request" ? body : "{}",
      },
    },
  });
  const responseBody = source === "request" ? "{}" : body;
  events.ingest({
    method: "Network.responseReceived",
    params: {
      requestId: "request-1",
      response: {
        url: `${ORIGIN}/api`,
        status: 200,
        mimeType: "application/json",
        headers: {},
      },
    },
  });
  events.ingest({
    method: "Network.loadingFinished",
    params: {
      requestId: "request-1",
      encodedDataLength: Buffer.byteLength(responseBody),
    },
  });
  expect(events.responseBodyRequestIds()).toEqual(["request-1"]);
  events.ingestResponseBody("request-1", {
    body:
      source === "base64 response"
        ? Buffer.from(responseBody).toString("base64")
        : responseBody,
    base64Encoded: source === "base64 response",
  });
  // Match CdpPageCapture's normalized projection, retaining actual event records
  // and attach-limited coverage; unrelated page sections are empty fixtures.
  return webPageInspectionSchema.parse({
    browser: {
      product: "fixture",
      protocol_version: "1.3",
      revision: "fixture",
      user_agent: "fixture",
      js_version: "fixture",
    },
    target: {
      target_id: "page-1",
      type: "page",
      title: "fixture",
      url: ORIGIN,
      origin: ORIGIN,
      attached: true,
    },
    capture_window: {
      started_at: CAPTURE_TIME,
      ended_at: CAPTURE_TIME,
      observation_ms: 0,
    },
    completeness: events.completeness.snapshot(),
    frames: [],
    dom: { total_nodes: 0, nodes: [] },
    accessibility: {
      total_nodes: 0,
      text_capture: {
        status: "not_approved",
        retained_bytes: 0,
        excluded_fields: 0,
      },
      nodes: [],
    },
    scripts: { total: 0, items: [] },
    resources: [],
    network: {
      requests: [...events.network.values()],
      websocket_events: events.websockets,
      coverage_started_at: CAPTURE_TIME,
      prior_activity_available: false,
    },
    console: {
      events: events.console,
      coverage_started_at: CAPTURE_TIME,
      prior_activity_available: false,
    },
    workers: [],
    metadata: {
      responses: events.responseMetadata,
      dom_urls: [],
      agent_hints: events.agentHints,
      excluded_dom_urls: 0,
      headers_allowlisted: true,
    },
    storage: {
      origin: ORIGIN,
      usage_bytes: null,
      quota_bytes: null,
      local_storage_keys: [],
      session_storage_keys: [],
      indexed_db_names: [],
      cache_names: [],
      values_redacted: true,
    },
    limitations: [],
  });
};
