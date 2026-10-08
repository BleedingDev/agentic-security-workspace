import { describe, expect, it } from "vitest";

import { HopperApplicationLauncher } from "../../../../src/hopper/BridgeLauncher.js";

describe("Hopper target leases in the launcher", () => {
  it("reports an active target owner without launching another document", async () => {
    let leaseChecks = 0;
    const launcher = new HopperApplicationLauncher(
      {
        launcherPath:
          "/Applications/Hopper Disassembler.app/Contents/MacOS/hopper",
        targetPath: "/target",
        targetKind: "executable",
        loaderArgs: [],
        bridgeScriptPath: "/rea/hopper_bridge.py",
        launchMode: "native",
      },
      {
        platform: "darwin",
        acquireTargetLease: async () => {
          leaseChecks += 1;
          return {
            acquired: false,
            owner: { runId: "owning-session", processId: 17 },
          };
        },
      },
    );

    const result = await launcher.launch({
      directory: "/tmp/rea-launch-test",
      socketPath: "/tmp/rea-launch-test/bridge.sock",
      token: "token",
      runId: "new-session",
    });

    expect(leaseChecks).toBe(1);
    expect(result).toMatchObject({
      ok: false,
      error: {
        _tag: "HopperStartError",
        ownerRunId: "owning-session",
        userMessage: expect.stringContaining("owning-session"),
      },
    });
  });

  it("releases a target lease when bridge setup fails", async () => {
    let releases = 0;
    const launcher = new HopperApplicationLauncher(
      {
        launcherPath: "/unused/hopper",
        targetPath: "/target",
        targetKind: "executable",
        loaderArgs: [],
        bridgeScriptPath: "/rea/hopper_bridge.py",
        launchMode: "native",
      },
      {
        platform: "darwin",
        acquireTargetLease: async () => ({
          acquired: true,
          lease: { release: async () => void (releases += 1) },
        }),
      },
    );

    const result = await launcher.launch({
      directory: "/missing/rea-launch-test",
      socketPath: "/missing/rea-launch-test/bridge.sock",
      token: "token",
      runId: "failed-session",
    });

    expect(result.ok).toBe(false);
    expect(releases).toBe(1);
  });
});
