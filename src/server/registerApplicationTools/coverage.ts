import type { McpServer } from "@modelcontextprotocol/server";

import { evaluateReconstructionCoverage } from "../../application/ReconstructionCoverageService.js";
import { applicationToolContract } from "../../contracts/applicationToolContracts.js";
import { logToolExecution } from "../toolLogging.js";
import { toolRegistrationOptions } from "../toolRegistrationOptions.js";
import { toCallToolResult } from "../toolResult.js";
import type { ApplicationToolRegistration } from "./types.js";

const contract = applicationToolContract("evaluate_reconstruction_coverage");

/** Register the inline fail-closed reconstruction coverage evaluator. */
export const registerCoverageTools = (
  server: McpServer,
  options: ApplicationToolRegistration,
): void => {
  server.registerTool(
    contract.name,
    toolRegistrationOptions(contract),
    async (input) => {
      const result = await logToolExecution(options.logger, contract.name, () =>
        Promise.resolve(evaluateReconstructionCoverage(input)),
      );
      return toCallToolResult(result, contract);
    },
  );
};
