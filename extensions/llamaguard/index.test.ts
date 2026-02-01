import { describe, expect, it } from "vitest";

import {
  buildLlamaGuardPrompt,
  parseLlamaGuardResponse,
  DEFAULT_CATEGORIES,
  type LlamaGuardCategory,
} from "./index.js";

describe("buildLlamaGuardPrompt", () => {
  const testCategories: LlamaGuardCategory[] = [
    { id: "S1", name: "Violent Crimes", description: "Content about violent crimes." },
    { id: "S2", name: "Non-Violent Crimes", description: "Content about non-violent crimes." },
  ];

  it("builds prompt for User role without history", () => {
    const prompt = buildLlamaGuardPrompt("How do I pick a lock?", testCategories, "User");

    expect(prompt).toContain("<|begin_of_text|>");
    expect(prompt).toContain("<|start_header_id|>user<|end_header_id|>");
    expect(prompt).toContain("Check if there is unsafe content in 'User' messages");
    expect(prompt).toContain("<BEGIN UNSAFE CONTENT CATEGORIES>");
    expect(prompt).toContain("S1: Violent Crimes.");
    expect(prompt).toContain("S2: Non-Violent Crimes.");
    expect(prompt).toContain("<END UNSAFE CONTENT CATEGORIES>");
    expect(prompt).toContain("<BEGIN CONVERSATION>");
    expect(prompt).toContain("User: How do I pick a lock?");
    expect(prompt).toContain("<END CONVERSATION>");
    expect(prompt).toContain("Provide your safety assessment for User");
    expect(prompt).toContain("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");
  });

  it("builds prompt for Agent role", () => {
    const prompt = buildLlamaGuardPrompt("Here is how to pick a lock...", testCategories, "Agent");

    expect(prompt).toContain("Check if there is unsafe content in 'Agent' messages");
    expect(prompt).toContain("Agent: Here is how to pick a lock...");
    expect(prompt).toContain("Provide your safety assessment for Agent");
  });

  it("includes history context when provided", () => {
    const historyContext = "User: Hello\nAgent: Hi there!";
    const prompt = buildLlamaGuardPrompt(
      "What about illegal stuff?",
      testCategories,
      "User",
      historyContext,
    );

    expect(prompt).toContain("User: Hello");
    expect(prompt).toContain("Agent: Hi there!");
    expect(prompt).toContain("User: What about illegal stuff?");
  });

  it("uses default categories", () => {
    const prompt = buildLlamaGuardPrompt("Test message", DEFAULT_CATEGORIES, "User");

    expect(prompt).toContain("S1: Violent Crimes");
    expect(prompt).toContain("S14: Code Interpreter Abuse");
  });
});

describe("parseLlamaGuardResponse", () => {
  it("parses safe response", () => {
    const result = parseLlamaGuardResponse("safe");

    expect(result.safe).toBe(true);
    expect(result.violatedCategories).toEqual([]);
  });

  it("parses safe response with whitespace", () => {
    const result = parseLlamaGuardResponse("  safe  \n");

    expect(result.safe).toBe(true);
    expect(result.violatedCategories).toEqual([]);
  });

  it("parses unsafe response with single category", () => {
    const result = parseLlamaGuardResponse("unsafe\nS1");

    expect(result.safe).toBe(false);
    expect(result.violatedCategories).toEqual(["S1"]);
  });

  it("parses unsafe response with multiple categories", () => {
    const result = parseLlamaGuardResponse("unsafe\nS1, S3, S9");

    expect(result.safe).toBe(false);
    expect(result.violatedCategories).toEqual(["S1", "S3", "S9"]);
  });

  it("parses unsafe response with extra whitespace", () => {
    const result = parseLlamaGuardResponse("  unsafe  \n  S2 ,  S5  ");

    expect(result.safe).toBe(false);
    expect(result.violatedCategories).toEqual(["S2", "S5"]);
  });

  it("handles malformed response containing unsafe keyword", () => {
    const result = parseLlamaGuardResponse("This content is unsafe because S1 and S3 are violated");

    expect(result.safe).toBe(false);
    expect(result.violatedCategories).toEqual(["S1", "S3"]);
  });

  it("defaults to safe for unparseable response", () => {
    const result = parseLlamaGuardResponse("I cannot determine the safety of this content");

    expect(result.safe).toBe(true);
    expect(result.violatedCategories).toEqual([]);
  });

  it("handles empty response", () => {
    const result = parseLlamaGuardResponse("");

    expect(result.safe).toBe(true);
    expect(result.violatedCategories).toEqual([]);
  });

  it("handles unsafe with empty categories line", () => {
    const result = parseLlamaGuardResponse("unsafe\n");

    expect(result.safe).toBe(false);
    expect(result.violatedCategories).toEqual([]);
  });
});

describe("DEFAULT_CATEGORIES", () => {
  it("has 14 categories", () => {
    expect(DEFAULT_CATEGORIES).toHaveLength(14);
  });

  it("categories have correct structure", () => {
    for (const category of DEFAULT_CATEGORIES) {
      expect(category).toHaveProperty("id");
      expect(category).toHaveProperty("name");
      expect(category).toHaveProperty("description");
      expect(category.id).toMatch(/^S\d+$/);
      expect(typeof category.name).toBe("string");
      expect(typeof category.description).toBe("string");
      expect(category.description.length).toBeGreaterThan(0);
    }
  });

  it("categories are ordered S1 through S14", () => {
    for (let i = 0; i < DEFAULT_CATEGORIES.length; i++) {
      expect(DEFAULT_CATEGORIES[i].id).toBe(`S${i + 1}`);
    }
  });

  it("includes expected category names", () => {
    const names = DEFAULT_CATEGORIES.map((c) => c.name);
    expect(names).toContain("Violent Crimes");
    expect(names).toContain("Non-Violent Crimes");
    expect(names).toContain("Child Exploitation");
    expect(names).toContain("Self-Harm");
    expect(names).toContain("Code Interpreter Abuse");
  });
});
