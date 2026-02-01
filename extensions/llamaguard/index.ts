/**
 * OpenClaw Llama Guard 3 Guardrails Plugin
 *
 * Provides guardrail functionality using Llama Guard 3 8B model via Ollama or
 * any OpenAI-compatible endpoint. Uses the built-in model provider system.
 */

import path from "node:path";

import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";

import type { OpenClawConfig } from "../../src/config/config.js";
import {
  type BaseStageConfig,
  type EmbeddedAgentResult,
  appendWarningToToolResult,
  buildToolCallSummary,
  cleanupTempDir,
  collectText,
  createGuardrailTempDir,
  extractMessagesContent,
  extractTextFromContent,
  extractToolResultText,
  generateSessionId,
  isStageEnabled,
  loadRunEmbeddedPiAgent,
  replaceToolResultWithWarning,
  resolveBlockMode,
  resolveStageConfig,
} from "../../src/plugins/guardrails-utils.js";

// ============================================================================
// Types
// ============================================================================

type LlamaGuardCategory = {
  id: string;
  name: string;
  description: string;
};

type LlamaGuardStageConfig = BaseStageConfig;

type LlamaGuardConfig = {
  enabled?: boolean;
  provider?: string;
  model?: string;
  authProfileId?: string;
  categories?: LlamaGuardCategory[];
  timeoutMs?: number;
  failOpen?: boolean;
  maxTokens?: number;
  stages?: {
    beforeRequest?: LlamaGuardStageConfig;
    beforeToolCall?: LlamaGuardStageConfig;
    afterToolCall?: LlamaGuardStageConfig;
    afterResponse?: LlamaGuardStageConfig;
  };
};

type LlamaGuardEvaluation = {
  safe: boolean;
  violatedCategories: string[];
};

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_PROVIDER = "ollama";
const DEFAULT_MODEL = "llama-guard3:8b";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_TOKENS = 100;

/**
 * Default Llama Guard 3 safety categories (S1-S14)
 * https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/MODEL_CARD.md
 */
const DEFAULT_CATEGORIES: LlamaGuardCategory[] = [
  {
    id: "S1",
    name: "Violent Crimes",
    description:
      "AI models should not create content that enables, encourages, or excuses the commission of violent crimes. Examples of violent crimes include, but are not limited to: unlawful violence toward people (e.g., terrorism, genocide, murder, hate crimes, child abuse, assault, battery, kidnapping), unlawful violence toward animals (e.g., animal abuse).",
  },
  {
    id: "S2",
    name: "Non-Violent Crimes",
    description:
      "AI models should not create content that enables, encourages, or excuses the commission of non-violent crimes. Examples of non-violent crimes include, but are not limited to: financial crimes (e.g., fraud, scams, money laundering), property crimes (e.g., burglary, robbery, arson, vandalism), drug crimes (e.g., creating combating substances, trafficking controlled substances), weapons crimes (e.g., producing combating weapons, unlawful sale/transfer of firearms), cyber crimes (e.g., hacking, spyware, phishing).",
  },
  {
    id: "S3",
    name: "Sex Crimes",
    description:
      "AI models should not create content that enables, encourages, or excuses the commission of sex crimes. Examples of sex crimes include, but are not limited to: human trafficking, sexual assault (e.g., rape), sexual harassment (e.g., groping), lewd conduct.",
  },
  {
    id: "S4",
    name: "Child Exploitation",
    description:
      "AI models should not create content that depicts child sexual abuse or exploits children in any way. This includes, but is not limited to: child sexual abuse material (CSAM), any content that sexualizes children, content that enables or encourages child abuse.",
  },
  {
    id: "S5",
    name: "Defamation",
    description:
      "AI models should not create content that is defamatory about a real, living person. Defamation is defined as a false statement of fact that damages the reputation of the person.",
  },
  {
    id: "S6",
    name: "Specialized Advice",
    description:
      "AI models should not create content that contains specialized financial, medical, or legal advice. This does not include general educational content on these topics.",
  },
  {
    id: "S7",
    name: "Privacy",
    description:
      "AI models should not create content that contains sensitive, personal information about private individuals that could lead to harm. This includes, but is not limited to: full names paired with, including but not limited to: phone numbers, SSN, address, email, bank account, driver's license, or passwords.",
  },
  {
    id: "S8",
    name: "Intellectual Property",
    description:
      "AI models should not create content that violates the intellectual property rights of third parties, such as reproducing copyrighted material verbatim, or creating content that infringes on trademarks or patents.",
  },
  {
    id: "S9",
    name: "Indiscriminate Weapons",
    description:
      "AI models should not create content that enables the creation of weapons of mass destruction. Examples include, but are not limited to: chemical weapons (e.g., nerve agents), biological weapons (e.g., anthrax), radiological weapons (e.g., dirty bombs), nuclear weapons (e.g., nuclear fission).",
  },
  {
    id: "S10",
    name: "Hate",
    description:
      "AI models should not create content that is hateful toward people on the basis of their protected characteristics. Protected characteristics include race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and serious disease. Severe disparagement and dehumanization are examples of hate speech.",
  },
  {
    id: "S11",
    name: "Self-Harm",
    description:
      "AI models should not create content that enables, encourages, or excuses acts of intentional self-harm. Examples include, but are not limited to: suicide, self-injury (e.g., cutting), disordered eating.",
  },
  {
    id: "S12",
    name: "Sexual Content",
    description:
      "AI models should not create depictions of nude adults or content that contains erotic descriptions or explicit depictions of sex acts.",
  },
  {
    id: "S13",
    name: "Elections",
    description:
      "AI models should not create content that contains factually incorrect information about electoral systems and processes, including the timing, location, or eligibility requirements for voting. It also includes content that could be used to materially discourage voting.",
  },
  {
    id: "S14",
    name: "Code Interpreter Abuse",
    description:
      "AI models should not create content that attempts to abuse code interpreters. Examples include, but are not limited to: writing malicious code for system exploitation, shell injection, or data exfiltration.",
  },
];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Build the Llama Guard 3 prompt following the official format.
 */
function buildLlamaGuardPrompt(
  content: string,
  categories: LlamaGuardCategory[],
  role: "User" | "Agent",
  historyContext?: string,
): string {
  const categoriesBlock = categories
    .map((cat) => `${cat.id}: ${cat.name}.\n${cat.description}`)
    .join("\n");

  const conversationBlock = historyContext
    ? `${historyContext}\n${role}: ${content}`
    : `${role}: ${content}`;

  return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in '${role}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
${categoriesBlock}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
${conversationBlock}
<END CONVERSATION>

Provide your safety assessment for ${role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>`;
}

/**
 * Parse Llama Guard response.
 * Expected format:
 *   safe
 * or:
 *   unsafe
 *   S1, S3
 */
function parseLlamaGuardResponse(response: string): LlamaGuardEvaluation {
  const lines = response.trim().split("\n");
  const firstLine = lines[0]?.toLowerCase().trim() ?? "";

  if (firstLine === "safe") {
    return { safe: true, violatedCategories: [] };
  }

  if (firstLine === "unsafe") {
    const categoriesLine = lines[1]?.trim() ?? "";
    const violatedCategories = categoriesLine
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    return { safe: false, violatedCategories };
  }

  // Fallback: if response contains "unsafe" anywhere, treat as unsafe
  if (response.toLowerCase().includes("unsafe")) {
    const match = response.match(/S\d+/g);
    return { safe: false, violatedCategories: match ?? [] };
  }

  // Default to safe if we can't parse
  return { safe: true, violatedCategories: [] };
}

async function callLlamaGuard(
  params: {
    cfg: LlamaGuardConfig;
    content: string;
    role: "User" | "Agent";
    historyContext?: string;
    apiConfig: OpenClawConfig;
  },
  api: OpenClawPluginApi,
): Promise<LlamaGuardEvaluation | null> {
  const provider = params.cfg.provider ?? DEFAULT_PROVIDER;
  const model = params.cfg.model ?? DEFAULT_MODEL;
  const categories = params.cfg.categories ?? DEFAULT_CATEGORIES;
  const timeoutMs = params.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxTokens = params.cfg.maxTokens ?? DEFAULT_MAX_TOKENS;

  const prompt = buildLlamaGuardPrompt(
    params.content,
    categories,
    params.role,
    params.historyContext,
  );

  let tmpDir: string | null = null;
  try {
    tmpDir = await createGuardrailTempDir("llamaguard");
    const sessionId = generateSessionId("llamaguard");
    const sessionFile = path.join(tmpDir, "session.json");

    const runEmbeddedPiAgent = await loadRunEmbeddedPiAgent();

    const result = await runEmbeddedPiAgent({
      sessionId,
      sessionFile,
      workspaceDir: process.cwd(),
      config: params.apiConfig,
      prompt,
      timeoutMs,
      runId: sessionId,
      provider,
      model,
      authProfileId: params.cfg.authProfileId,
      authProfileIdSource: params.cfg.authProfileId ? "user" : "auto",
      streamParams: { maxTokens },
      disableTools: true,
    });

    const text = collectText((result as EmbeddedAgentResult).payloads);
    if (!text) {
      api.logger.warn("Llama Guard returned empty response");
      if (params.cfg.failOpen === false) {
        throw new Error("Llama Guard returned empty response");
      }
      return null;
    }

    return parseLlamaGuardResponse(text);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    api.logger.warn(`Llama Guard call failed: ${message}`);
    if (params.cfg.failOpen === false) {
      throw err;
    }
    return null;
  } finally {
    await cleanupTempDir(tmpDir);
  }
}

function formatViolationMessage(params: {
  evaluation: LlamaGuardEvaluation;
  location: string;
  categories: LlamaGuardCategory[];
}): string {
  const violatedCategories = params.evaluation.violatedCategories;
  const categoryMap = new Map(params.categories.map((c) => [c.id, c.name]));

  const categoryNames = violatedCategories
    .map((id) => {
      const name = categoryMap.get(id);
      return name ? `${id} (${name})` : id;
    })
    .join(", ");

  const messageParts = [
    `Sorry, I can't help with that. The ${params.location} was flagged as potentially unsafe by the Llama Guard safety system.`,
  ];

  if (categoryNames) {
    messageParts.push(`Violated categories: ${categoryNames}.`);
  }

  return messageParts.join(" ");
}

// ============================================================================
// Plugin Definition
// ============================================================================

const llamaguardPlugin = {
  id: "llamaguard",
  name: "Llama Guard 3 Guardrails",
  description: "Content safety guardrails via Llama Guard 3 8B",
  configSchema: emptyPluginConfigSchema(),

  register(api: OpenClawPluginApi) {
    const cfg = api.pluginConfig as LlamaGuardConfig | undefined;
    if (!cfg || cfg.enabled === false) {
      api.logger.debug?.("Llama Guard guardrails disabled or not configured");
      return;
    }

    const categories = cfg.categories ?? DEFAULT_CATEGORIES;

    api.logger.info(
      `Llama Guard guardrails enabled (provider: ${cfg.provider ?? DEFAULT_PROVIDER}, model: ${cfg.model ?? DEFAULT_MODEL})`,
    );

    // Register before_request hook
    api.on(
      "before_request",
      async (event) => {
        const stageCfg = resolveStageConfig(cfg.stages, "before_request");
        if (!isStageEnabled(stageCfg)) {
          return;
        }
        const prompt = event.prompt.trim();
        if (!prompt) {
          return;
        }

        const includeHistory = stageCfg?.includeHistory !== false;
        const historyContext = includeHistory ? extractMessagesContent(event.messages) : undefined;

        let evaluation: LlamaGuardEvaluation | null = null;
        try {
          evaluation = await callLlamaGuard(
            {
              cfg,
              content: prompt,
              role: "User",
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockResponse: "Request blocked because Llama Guard guardrail failed.",
          };
        }

        if (!evaluation) {
          return;
        }

        if (evaluation.safe) {
          return;
        }

        if (stageCfg?.mode === "monitor") {
          api.logger.warn(
            `[monitor] Llama Guard flagged input: ${evaluation.violatedCategories.join(", ")}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "input query",
          categories,
        });
        return {
          block: true,
          blockResponse: message,
        };
      },
      { priority: 50 },
    );

    // Register before_tool_call hook
    api.on(
      "before_tool_call",
      async (event) => {
        const stageCfg = resolveStageConfig(cfg.stages, "before_tool_call");
        if (!isStageEnabled(stageCfg)) {
          return;
        }

        const toolSummary = buildToolCallSummary(event.toolName, event.toolCallId, event.params);
        const includeHistory = stageCfg?.includeHistory !== false;
        const historyContext = includeHistory ? extractMessagesContent(event.messages) : undefined;

        let evaluation: LlamaGuardEvaluation | null = null;
        try {
          evaluation = await callLlamaGuard(
            {
              cfg,
              content: toolSummary,
              role: "Agent",
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockReason: "Tool call blocked because Llama Guard guardrail failed.",
          };
        }

        if (!evaluation) {
          return;
        }

        if (evaluation.safe) {
          return;
        }

        if (stageCfg?.mode === "monitor") {
          api.logger.warn(
            `[monitor] Llama Guard flagged tool call ${event.toolName}: ${evaluation.violatedCategories.join(", ")}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "tool call request",
          categories,
        });
        return {
          block: true,
          blockReason: message,
        };
      },
      { priority: 50 },
    );

    // Register after_tool_call hook
    api.on(
      "after_tool_call",
      async (event) => {
        const stageCfg = resolveStageConfig(cfg.stages, "after_tool_call");
        if (!isStageEnabled(stageCfg)) {
          return;
        }

        const toolText = extractToolResultText(event.result).trim();
        if (!toolText) {
          return;
        }

        const includeHistory = stageCfg?.includeHistory !== false;
        const historyContext = includeHistory ? extractMessagesContent(event.messages) : undefined;

        let evaluation: LlamaGuardEvaluation | null = null;
        try {
          evaluation = await callLlamaGuard(
            {
              cfg,
              content: toolText,
              role: "Agent",
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            result: replaceToolResultWithWarning(
              event.result,
              "Tool result blocked because Llama Guard guardrail failed.",
            ),
          };
        }

        if (!evaluation) {
          return;
        }

        if (evaluation.safe) {
          return;
        }

        if (stageCfg?.mode === "monitor") {
          api.logger.warn(
            `[monitor] Llama Guard flagged tool result ${event.toolName}: ${evaluation.violatedCategories.join(", ")}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "tool response",
          categories,
        });
        const blockMode = resolveBlockMode("after_tool_call", stageCfg);
        return {
          block: true,
          result:
            blockMode === "append"
              ? appendWarningToToolResult(event.result, message)
              : replaceToolResultWithWarning(event.result, message),
        };
      },
      { priority: 50 },
    );

    // Register after_response hook
    api.on(
      "after_response",
      async (event) => {
        const stageCfg = resolveStageConfig(cfg.stages, "after_response");
        if (!isStageEnabled(stageCfg)) {
          return;
        }

        const assistantText =
          event.assistantTexts.join("\n").trim() ||
          (event.lastAssistant
            ? extractTextFromContent((event.lastAssistant as AssistantMessage).content).trim()
            : "");
        if (!assistantText) {
          return;
        }

        const includeHistory = stageCfg?.includeHistory !== false;
        const historyContext = includeHistory ? extractMessagesContent(event.messages) : undefined;

        let evaluation: LlamaGuardEvaluation | null = null;
        try {
          evaluation = await callLlamaGuard(
            {
              cfg,
              content: assistantText,
              role: "Agent",
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockResponse: "Response blocked because Llama Guard guardrail failed.",
          };
        }

        if (!evaluation) {
          return;
        }

        if (evaluation.safe) {
          return;
        }

        if (stageCfg?.mode === "monitor") {
          api.logger.warn(
            `[monitor] Llama Guard flagged response: ${evaluation.violatedCategories.join(", ")}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "model response",
          categories,
        });
        const blockMode = resolveBlockMode("after_response", stageCfg);
        if (blockMode === "append") {
          return {
            assistantTexts: [...event.assistantTexts, message],
          };
        }
        return {
          block: true,
          blockResponse: message,
        };
      },
      { priority: 50 },
    );
  },
};

export default llamaguardPlugin;

// Export types for testing
export type { LlamaGuardConfig, LlamaGuardCategory, LlamaGuardStageConfig, LlamaGuardEvaluation };
export { buildLlamaGuardPrompt, parseLlamaGuardResponse, DEFAULT_CATEGORIES };
