import { it, expect, describe, vi } from "vitest";

describe("getEncoding", () => {
  it("rejects encodings that are not in the allowlist", async () => {
    const { getEncoding } = await import("../tiktoken.js");
    await expect(
      // Cast through unknown because the type system already forbids this, but
      // runtime callers may pass arbitrary strings.
      getEncoding("../../../../etc/passwd" as unknown as never)
    ).rejects.toThrow(/Invalid encoding/);
  });

  it("does not perform a network request for an invalid encoding", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const { getEncoding } = await import("../tiktoken.js");
    await expect(
      getEncoding("not-a-real-encoding" as unknown as never)
    ).rejects.toThrow(/Invalid encoding/);
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });
});
