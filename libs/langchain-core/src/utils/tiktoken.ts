import {
  Tiktoken,
  TiktokenEncoding,
  TiktokenModel,
  getEncodingNameForModel,
} from "js-tiktoken/lite";
import { AsyncCaller } from "./async_caller.js";

const cache: Record<string, Promise<Tiktoken>> = {};

const caller = /* #__PURE__ */ new AsyncCaller({});

/**
 * Allowlist of encoding names supported by js-tiktoken. The encoding name is
 * interpolated into the URL used to fetch the BPE ranks, so it must be
 * validated against this list to prevent path traversal / SSRF if an
 * untrusted value is ever passed in.
 */
const VALID_ENCODINGS: ReadonlySet<TiktokenEncoding> = /* #__PURE__ */ new Set([
  "gpt2",
  "r50k_base",
  "p50k_base",
  "p50k_edit",
  "cl100k_base",
  "o200k_base",
]);

export async function getEncoding(encoding: TiktokenEncoding) {
  if (!VALID_ENCODINGS.has(encoding)) {
    throw new Error(
      `Invalid encoding "${encoding}". Must be one of: ${[
        ...VALID_ENCODINGS,
      ].join(", ")}`
    );
  }

  if (!(encoding in cache)) {
    cache[encoding] = caller
      .fetch(`https://tiktoken.pages.dev/js/${encoding}.json`)
      .then((res) => res.json())
      .then((data) => new Tiktoken(data))
      .catch((e) => {
        delete cache[encoding];
        throw e;
      });
  }

  return await cache[encoding];
}

export async function encodingForModel(model: TiktokenModel) {
  return getEncoding(getEncodingNameForModel(model));
}
