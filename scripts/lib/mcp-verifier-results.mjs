/** Read the textual JSON projection from one MCP tool call. */
export const mcpTextValue = (result) => {
  const text = result.content?.find((item) => item.type === "text")?.text;
  if (typeof text !== "string") throw new Error("Tool result omitted text");
  return text;
};

const jsonValue = (result) => JSON.parse(mcpTextValue(result));

/** Return the compact result projection from one successful MCP tool call. */
export const requireMcpResult = (result, operation) => {
  if (result.isError === true) {
    throw new Error(`${operation} failed: ${mcpTextValue(result)}`);
  }
  const value = jsonValue(result);
  if (value === null || typeof value !== "object" || !("result" in value)) {
    throw new Error(`${operation} omitted its result projection`);
  }
  return value.result;
};

/** Verify provider identity directly from the tool result. */
export const requireEvidenceProvider = (result, operation, providerId) => {
  const evidence = jsonValue(result);
  if (
    evidence?.evidence?.provider?.id !== providerId ||
    evidence?.evidence?.analysis_profile?.provider?.id !== providerId ||
    typeof evidence.evidence.analysis_profile.provider.version !== "string"
  ) {
    throw new Error(`${operation} omitted its concrete provider provenance`);
  }
};

/** Verify composed workflow and upstream provider provenance inline. */
export const requireWorkflowEvidenceProvider = (
  result,
  operation,
  expected,
) => {
  const evidence = jsonValue(result);
  const profile = evidence?.evidence?.analysis_profile;
  const upstream = profile?.parameters?.upstream_analysis_profile;
  if (
    evidence?.evidence?.provider?.id !== expected.workflowProviderId ||
    profile?.provider?.id !== expected.workflowProviderId ||
    upstream?.provider?.id !== expected.upstreamProviderId ||
    typeof upstream.provider.version !== "string"
  ) {
    throw new Error(
      `${operation} omitted its composed workflow or upstream provenance`,
    );
  }
};
