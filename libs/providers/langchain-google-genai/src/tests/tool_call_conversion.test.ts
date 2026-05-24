import { test, expect, describe } from "vitest";
import { AIMessage } from "@langchain/core/messages";
import { convertBaseMessagesToContent } from "../utils/common.js";
import { convertToolsToGenAI } from "../utils/tools.js";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

test("converts standard tool_call content blocks to Google functionCall format", () => {
  // Create AIMessage with standard tool_call content block
  const aiMessage = new AIMessage({
    contentBlocks: [
      {
        type: "tool_call",
        id: "call_123",
        name: "calculator",
        args: {
          operation: "add",
          number1: 2,
          number2: 3,
        },
      },
    ],
  });

  // Convert to Google GenAI format
  const result = convertBaseMessagesToContent(
    [aiMessage],
    false, // isMultimodalModel
    undefined,
    "gemini-1.5-flash"
  );

  // Verify correct conversion
  expect(result).toBeDefined();
  expect(result.length).toBe(1);
  const part = result[0].parts[0];

  expect(part.functionCall).toBeDefined();
  expect(part.functionCall?.name).toBe("calculator");
  expect(part.functionCall?.args).toEqual({
    operation: "add",
    number1: 2,
    number2: 3,
  });
});

describe("Gemini Tool Schema Validation - Empty String in Enum", () => {
  test("should throw descriptive error for empty string in z.enum", () => {
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash-exp",
      apiKey: "fake-api-key",
    });

    const schema = z.object({
      status: z.enum(["", "active", "inactive"]).describe("Status value"),
    });

    const crashTool = tool(async (_input: z.infer<typeof schema>) => "ok", {
      name: "crash_tool",
      description: "This tool will crash Gemini SDK",
      schema: schema,
    });

    expect(() => model.bindTools([crashTool])).toThrow(
      /Invalid enum: empty string not allowed/
    );
  });

  test("should throw descriptive error for empty string in z.nativeEnum", () => {
    enum TestEnum {
      A = "",
      B = "active",
      C = "inactive",
    }

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash-exp",
      apiKey: "fake-api-key",
    });

    const schema = z.object({
      status: z.nativeEnum(TestEnum).describe("Status value"),
    });

    const crashTool = tool(async (_input: z.infer<typeof schema>) => "ok", {
      name: "crash_tool",
      description: "This tool will crash Gemini SDK",
      schema: schema,
    });

    expect(() => model.bindTools([crashTool])).toThrow(
      /Invalid enum: empty string not allowed/
    );
  });
});

describe("Mixing native Google tools with LangChain tools", () => {
  test("should handle codeExecution mixed with LangChain tool", () => {
    const myTool = tool(async ({ query }) => `Result for ${query}`, {
      name: "search",
      description: "Search for information",
      schema: z.object({ query: z.string() }),
    });

    const result = convertToolsToGenAI([myTool, { codeExecution: {} }]);

    expect(result.tools).toHaveLength(2);

    const codeExecTool = result.tools.find(
      (t) => "codeExecution" in t
    );
    expect(codeExecTool).toBeDefined();

    const funcDeclTool = result.tools.find(
      (t) => "functionDeclarations" in t
    );
    expect(funcDeclTool).toBeDefined();
    expect(funcDeclTool!.functionDeclarations).toHaveLength(1);
    expect(funcDeclTool!.functionDeclarations![0].name).toBe("search");
  });

  test("should handle googleSearchRetrieval mixed with LangChain tool", () => {
    const myTool = tool(async ({ query }) => `Result for ${query}`, {
      name: "lookup",
      description: "Look up data",
      schema: z.object({ query: z.string() }),
    });

    const result = convertToolsToGenAI([
      { googleSearchRetrieval: {} },
      myTool,
    ]);

    expect(result.tools).toHaveLength(2);

    const searchTool = result.tools.find(
      (t) => "googleSearchRetrieval" in t
    );
    expect(searchTool).toBeDefined();

    const funcDeclTool = result.tools.find(
      (t) => "functionDeclarations" in t
    );
    expect(funcDeclTool).toBeDefined();
    expect(funcDeclTool!.functionDeclarations![0].name).toBe("lookup");
  });

  test("should handle googleSearch mixed with LangChain tool", () => {
    const myTool = tool(async ({ query }) => `Result for ${query}`, {
      name: "custom_tool",
      description: "A custom tool",
      schema: z.object({ query: z.string() }),
    });

    const result = convertToolsToGenAI([
      myTool,
      { googleSearch: {} },
    ]);

    expect(result.tools).toHaveLength(2);

    const searchTool = result.tools.find(
      (t) => "googleSearch" in t
    );
    expect(searchTool).toBeDefined();

    const funcDeclTool = result.tools.find(
      (t) => "functionDeclarations" in t
    );
    expect(funcDeclTool).toBeDefined();
    expect(funcDeclTool!.functionDeclarations![0].name).toBe("custom_tool");
  });

  test("should handle multiple native tools with multiple LangChain tools", () => {
    const tool1 = tool(async ({ a }) => String(a), {
      name: "tool_a",
      description: "Tool A",
      schema: z.object({ a: z.string() }),
    });
    const tool2 = tool(async ({ b }) => String(b), {
      name: "tool_b",
      description: "Tool B",
      schema: z.object({ b: z.number() }),
    });

    const result = convertToolsToGenAI([
      tool1,
      { codeExecution: {} },
      tool2,
      { googleSearch: {} },
    ]);

    expect(result.tools).toHaveLength(3);

    const codeExecTool = result.tools.find(
      (t) => "codeExecution" in t
    );
    expect(codeExecTool).toBeDefined();

    const searchTool = result.tools.find(
      (t) => "googleSearch" in t
    );
    expect(searchTool).toBeDefined();

    const funcDeclTool = result.tools.find(
      (t) => "functionDeclarations" in t
    );
    expect(funcDeclTool).toBeDefined();
    expect(funcDeclTool!.functionDeclarations).toHaveLength(2);
    expect(funcDeclTool!.functionDeclarations![0].name).toBe("tool_a");
    expect(funcDeclTool!.functionDeclarations![1].name).toBe("tool_b");
  });

  test("should handle native-only tools", () => {
    const result = convertToolsToGenAI([
      { codeExecution: {} },
      { googleSearch: {} },
    ]);

    expect(result.tools).toHaveLength(2);
    expect(result.tools[0]).toEqual({ codeExecution: {} });
    expect(result.tools[1]).toEqual({ googleSearch: {} });
  });

  test("should preserve existing functionDeclarations when mixing", () => {
    const result = convertToolsToGenAI([
      {
        functionDeclarations: [
          { name: "existing_func", description: "Already declared" },
        ],
      },
      { codeExecution: {} },
    ]);

    expect(result.tools).toHaveLength(2);

    const funcDeclTool = result.tools.find(
      (t) => "functionDeclarations" in t
    );
    expect(funcDeclTool).toBeDefined();
    expect(funcDeclTool!.functionDeclarations![0].name).toBe("existing_func");

    const codeExecTool = result.tools.find(
      (t) => "codeExecution" in t
    );
    expect(codeExecTool).toBeDefined();
  });
});
