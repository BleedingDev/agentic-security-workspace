import type { AppConfig } from "../config.js";

export interface RuntimeState {
  currentConfig: AppConfig;
}

export const createRuntimeState = (config: AppConfig): RuntimeState => ({
  currentConfig: config,
});
