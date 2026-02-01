/**
 * OpenClaw GPT-OSS-Safeguard Guardrails Plugin
 *
 * Provides guardrail functionality using GPT-OSS-Safeguard model via Ollama or
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

type SafeguardStageConfig = BaseStageConfig;

type SafeguardConfig = {
  enabled?: boolean;
  provider?: string;
  model?: string;
  authProfileId?: string;
  policy?: string;
  reasoningEffort?: "low" | "medium" | "high";
  outputFormat?: "binary" | "json" | "rich";
  timeoutMs?: number;
  failOpen?: boolean;
  maxTokens?: number;
  stages?: {
    beforeRequest?: SafeguardStageConfig;
    beforeToolCall?: SafeguardStageConfig;
    afterToolCall?: SafeguardStageConfig;
    afterResponse?: SafeguardStageConfig;
  };
};

type SafeguardEvaluation = {
  safe: boolean;
  violation?: boolean;
  policyCategory?: string;
  rationale?: string;
  confidence?: string;
};

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_PROVIDER = "ollama";
const DEFAULT_MODEL = "openai/gpt-oss-safeguard-120b";
const DEFAULT_REASONING_EFFORT = "medium";
const DEFAULT_OUTPUT_FORMAT = "json";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_TOKENS = 500;

const DEFAULT_POLICY = `You are a safety classifier. Evaluate if the content violates any of these policies:
1. Violent or harmful content
2. Hate speech or discrimination
3. Sexual content involving minors
4. Instructions for illegal activities
5. Personal information exposure

Classify as violation (1) or safe (0).`;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Build the GPT-OSS-Safeguard prompt.
 * Uses standard system/user message format with policy and output format instructions.
 */
function buildSafeguardPrompt(
  content: string,
  policy: string,
  reasoningEffort: string,
  outputFormat: string,
): { systemPrompt: string; userPrompt: string } {
  const outputInstructions =
    outputFormat === "binary"
      ? "Respond with exactly 0 (safe) or 1 (violation)."
      : outputFormat === "json"
        ? 'Respond with JSON: {"violation": 0|1, "policy_category": "category name or null"}'
        : 'Respond with JSON: {"violation": 0|1, "policy_category": "category name or null", "confidence": "low|medium|high", "rationale": "brief explanation"}';

  const systemPrompt = `${policy}

Reasoning effort: ${reasoningEffort}
${outputInstructions}`;

  const userPrompt = `Evaluate the following content:\n\n${content}`;

  return { systemPrompt, userPrompt };
}

/**
 * Parse GPT-OSS-Safeguard response.
 * Handles binary (0/1), JSON, and rich JSON formats.
 */
function parseSafeguardResponse(response: string, outputFormat: string): SafeguardEvaluation {
  const trimmed = response.trim();

  // Binary format: just "0" or "1"
  if (outputFormat === "binary") {
    const violation = trimmed === "1" || trimmed.startsWith("1");
    return { safe: !violation, violation };
  }

  // JSON formats
  try {
    // Try to extract JSON from the response (handles markdown code blocks)
    let jsonStr = trimmed;
    const jsonMatch = trimmed.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      jsonStr = jsonMatch[0];
    }

    const parsed = JSON.parse(jsonStr) as Record<string, unknown>;
    const violation = parsed.violation === 1 || parsed.violation === true;
    return {
      safe: !violation,
      violation,
      policyCategory:
        typeof parsed.policy_category === "string" ? parsed.policy_category : undefined,
      rationale: typeof parsed.rationale === "string" ? parsed.rationale : undefined,
      confidence: typeof parsed.confidence === "string" ? parsed.confidence : undefined,
    };
  } catch {
    // Fallback: check for violation indicators
    const hasViolation = /violation["']?\s*:\s*(1|true)/i.test(trimmed) || trimmed === "1";
    return { safe: !hasViolation, violation: hasViolation };
  }
}

async function callSafeguard(
  params: {
    cfg: SafeguardConfig;
    content: string;
    historyContext?: string;
    apiConfig: OpenClawConfig;
  },
  api: OpenClawPluginApi,
): Promise<SafeguardEvaluation | null> {
  const provider = params.cfg.provider ?? DEFAULT_PROVIDER;
  const model = params.cfg.model ?? DEFAULT_MODEL;
  const policy = params.cfg.policy ?? DEFAULT_POLICY;
  const reasoningEffort = params.cfg.reasoningEffort ?? DEFAULT_REASONING_EFFORT;
  const outputFormat = params.cfg.outputFormat ?? DEFAULT_OUTPUT_FORMAT;
  const timeoutMs = params.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxTokens = params.cfg.maxTokens ?? DEFAULT_MAX_TOKENS;

  // Include history context in the content if provided
  const fullContent = params.historyContext
    ? `${params.historyContext}\n\nCurrent content to evaluate:\n${params.content}`
    : params.content;

  const { systemPrompt, userPrompt } = buildSafeguardPrompt(
    fullContent,
    policy,
    reasoningEffort,
    outputFormat,
  );

  // Combine system prompt and user prompt for the embedded agent
  const prompt = `${systemPrompt}\n\n${userPrompt}`;

  let tmpDir: string | null = null;
  try {
    tmpDir = await createGuardrailTempDir("safeguard");
    const sessionId = generateSessionId("safeguard");
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
      api.logger.warn("GPT-OSS-Safeguard returned empty response");
      if (params.cfg.failOpen === false) {
        throw new Error("GPT-OSS-Safeguard returned empty response");
      }
      return null;
    }

    return parseSafeguardResponse(text, outputFormat);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    api.logger.warn(`GPT-OSS-Safeguard call failed: ${message}`);
    if (params.cfg.failOpen === false) {
      throw err;
    }
    return null;
  } finally {
    await cleanupTempDir(tmpDir);
  }
}

function formatViolationMessage(params: {
  evaluation: SafeguardEvaluation;
  location: string;
}): string {
  const messageParts = [
    `Sorry, I can't help with that. The ${params.location} was flagged as potentially unsafe by the GPT-OSS-Safeguard safety system.`,
  ];

  if (params.evaluation.policyCategory) {
    messageParts.push(`Policy category: ${params.evaluation.policyCategory}.`);
  }

  if (params.evaluation.rationale) {
    messageParts.push(`Reason: ${params.evaluation.rationale}`);
  }

  return messageParts.join(" ");
}

// ============================================================================
// Plugin Definition
// ============================================================================

const safeguardPlugin = {
  id: "gpt-oss-safeguard",
  name: "GPT-OSS-Safeguard Guardrails",
  description: "Content safety guardrails via GPT-OSS-Safeguard",
  configSchema: emptyPluginConfigSchema(),

  register(api: OpenClawPluginApi) {
    const cfg = api.pluginConfig as SafeguardConfig | undefined;
    if (!cfg || cfg.enabled === false) {
      api.logger.debug?.("GPT-OSS-Safeguard guardrails disabled or not configured");
      return;
    }

    api.logger.info(
      `GPT-OSS-Safeguard guardrails enabled (provider: ${cfg.provider ?? DEFAULT_PROVIDER}, model: ${cfg.model ?? DEFAULT_MODEL})`,
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

        let evaluation: SafeguardEvaluation | null = null;
        try {
          evaluation = await callSafeguard(
            {
              cfg,
              content: prompt,
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockResponse: "Request blocked because GPT-OSS-Safeguard guardrail failed.",
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
            `[monitor] GPT-OSS-Safeguard flagged input${evaluation.policyCategory ? `: ${evaluation.policyCategory}` : ""}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "input query",
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

        let evaluation: SafeguardEvaluation | null = null;
        try {
          evaluation = await callSafeguard(
            {
              cfg,
              content: toolSummary,
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockReason: "Tool call blocked because GPT-OSS-Safeguard guardrail failed.",
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
            `[monitor] GPT-OSS-Safeguard flagged tool call ${event.toolName}${evaluation.policyCategory ? `: ${evaluation.policyCategory}` : ""}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "tool call request",
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

        let evaluation: SafeguardEvaluation | null = null;
        try {
          evaluation = await callSafeguard(
            {
              cfg,
              content: toolText,
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
              "Tool result blocked because GPT-OSS-Safeguard guardrail failed.",
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
            `[monitor] GPT-OSS-Safeguard flagged tool result ${event.toolName}${evaluation.policyCategory ? `: ${evaluation.policyCategory}` : ""}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "tool response",
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

        let evaluation: SafeguardEvaluation | null = null;
        try {
          evaluation = await callSafeguard(
            {
              cfg,
              content: assistantText,
              historyContext,
              apiConfig: api.config,
            },
            api,
          );
        } catch {
          return {
            block: true,
            blockResponse: "Response blocked because GPT-OSS-Safeguard guardrail failed.",
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
            `[monitor] GPT-OSS-Safeguard flagged response${evaluation.policyCategory ? `: ${evaluation.policyCategory}` : ""}`,
          );
          return;
        }

        const message = formatViolationMessage({
          evaluation,
          location: "model response",
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

export default safeguardPlugin;

// Export types and functions for testing
export type { SafeguardConfig, SafeguardStageConfig, SafeguardEvaluation };
export {
  buildSafeguardPrompt,
  parseSafeguardResponse,
  DEFAULT_POLICY,
  DEFAULT_PROVIDER,
  DEFAULT_MODEL,
  DEFAULT_REASONING_EFFORT,
  DEFAULT_OUTPUT_FORMAT,
};
