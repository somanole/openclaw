/**
 * Shared utilities for guardrail plugins.
 *
 * Provides common types, content extraction, tool result manipulation,
 * and stage configuration helpers used across guardrail implementations.
 */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { AgentMessage, AgentToolResult } from "@mariozechner/pi-agent-core";

// ============================================================================
// Types
// ============================================================================

export type GuardrailStage =
  | "before_request"
  | "after_response"
  | "before_tool_call"
  | "after_tool_call";

export type BaseStageConfig = {
  enabled?: boolean;
  mode?: "block" | "monitor";
  blockMode?: "replace" | "append";
  includeHistory?: boolean;
};

export type RunEmbeddedPiAgentFn = (params: Record<string, unknown>) => Promise<unknown>;

export type EmbeddedAgentResult = {
  payloads?: Array<{ text?: string; isError?: boolean }>;
};

// ============================================================================
// Content Extraction
// ============================================================================

/**
 * Extract text content from various content formats (string, array of content blocks).
 */
export function extractTextFromContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const texts = content
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }
      const record = item as Record<string, unknown>;
      if (record.type && record.type !== "text") {
        return "";
      }
      return typeof record.text === "string" ? record.text : "";
    })
    .filter(Boolean);
  return texts.join("\n");
}

/**
 * Extract text from a tool result, falling back to JSON stringification.
 */
export function extractToolResultText(result: AgentToolResult<unknown>): string {
  if (result === null || result === undefined) {
    return "";
  }
  const contentText = extractTextFromContent(result?.content).trim();
  if (contentText) {
    return contentText;
  }
  if (result?.details !== undefined) {
    try {
      return JSON.stringify(result.details);
    } catch {
      return "";
    }
  }
  try {
    return JSON.stringify(result);
  } catch {
    return "";
  }
}

/**
 * Extract text content from conversation messages for context.
 */
export function extractMessagesContent(messages: AgentMessage[]): string {
  const parts: string[] = [];
  for (const message of messages) {
    const msgObj = message as { role?: unknown; content?: unknown };
    const role = msgObj.role;
    if (role !== "user" && role !== "assistant") {
      continue;
    }
    const content = extractTextFromContent(msgObj.content).trim();
    if (content) {
      const label = role === "user" ? "User" : "Agent";
      parts.push(`${label}: ${content}`);
    }
  }
  return parts.join("\n");
}

// ============================================================================
// Tool Result Manipulation
// ============================================================================

/**
 * Append a warning message to a tool result's content.
 */
export function appendWarningToToolResult(
  result: AgentToolResult<unknown>,
  warning: string,
): AgentToolResult<unknown> {
  const content = Array.isArray(result.content) ? [...result.content] : [];
  content.push({ type: "text", text: warning });
  return { ...result, content };
}

/**
 * Replace a tool result's content with a warning message.
 */
export function replaceToolResultWithWarning(
  result: AgentToolResult<unknown>,
  warning: string,
): AgentToolResult<unknown> {
  const baseDetails =
    result &&
    typeof result === "object" &&
    "details" in result &&
    (result as { details?: unknown }).details &&
    typeof (result as { details?: unknown }).details === "object"
      ? ((result as { details?: Record<string, unknown> }).details ?? {})
      : undefined;
  const details = baseDetails
    ? { ...baseDetails, guardrailWarning: warning }
    : { guardrailWarning: warning };
  return {
    content: [{ type: "text", text: warning }],
    details,
  };
}

/**
 * Build a JSON summary of a tool call for guardrail evaluation.
 */
export function buildToolCallSummary(
  toolName: string,
  toolCallId: string,
  params: unknown,
): string {
  try {
    return JSON.stringify({ tool: toolName, toolCallId, params });
  } catch {
    return toolName;
  }
}

// ============================================================================
// Stage Configuration
// ============================================================================

/**
 * Check if a guardrail stage is enabled.
 */
export function isStageEnabled(stage: BaseStageConfig | undefined): boolean {
  if (!stage) {
    return false;
  }
  return stage.enabled !== false;
}

/**
 * Resolve the block mode for a stage, defaulting to "append" for after_tool_call.
 */
export function resolveBlockMode(
  stage: GuardrailStage,
  stageCfg: BaseStageConfig | undefined,
): "replace" | "append" {
  if (stageCfg?.blockMode) {
    return stageCfg.blockMode;
  }
  if (stage === "after_tool_call") {
    return "append";
  }
  return "replace";
}

/**
 * Resolve stage configuration from a guardrail config object.
 */
export function resolveStageConfig<T extends BaseStageConfig>(
  stages:
    | {
        beforeRequest?: T;
        beforeToolCall?: T;
        afterToolCall?: T;
        afterResponse?: T;
      }
    | undefined,
  stage: GuardrailStage,
): T | undefined {
  if (!stages) {
    return undefined;
  }
  switch (stage) {
    case "before_request":
      return stages.beforeRequest;
    case "before_tool_call":
      return stages.beforeToolCall;
    case "after_tool_call":
      return stages.afterToolCall;
    case "after_response":
      return stages.afterResponse;
    default:
      return undefined;
  }
}

// ============================================================================
// Model Invocation Utilities (for local model-based guardrails)
// ============================================================================

/**
 * Load the embedded Pi agent runner function.
 * Tries source checkout first, then bundled install.
 */
export async function loadRunEmbeddedPiAgent(): Promise<RunEmbeddedPiAgentFn> {
  // Source checkout (tests/dev) - from src/plugins/ to src/agents/
  try {
    const mod = (await import("../agents/pi-embedded-runner.js")) as {
      runEmbeddedPiAgent?: unknown;
    };
    if (typeof mod.runEmbeddedPiAgent === "function") {
      return mod.runEmbeddedPiAgent as RunEmbeddedPiAgentFn;
    }
  } catch {
    // ignore
  }

  throw new Error("Internal error: runEmbeddedPiAgent not available");
}

/**
 * Collect text from embedded agent payloads.
 */
export function collectText(
  payloads: Array<{ text?: string; isError?: boolean }> | undefined,
): string {
  const texts = (payloads ?? [])
    .filter((p) => !p.isError && typeof p.text === "string")
    .map((p) => p.text ?? "");
  return texts.join("\n").trim();
}

/**
 * Create a temporary directory for guardrail sessions.
 */
export async function createGuardrailTempDir(prefix: string): Promise<string> {
  return fs.mkdtemp(path.join(os.tmpdir(), `openclaw-${prefix}-`));
}

/**
 * Clean up a temporary directory.
 */
export async function cleanupTempDir(tmpDir: string | null): Promise<void> {
  if (tmpDir) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

/**
 * Generate a unique session ID for guardrail calls.
 */
export function generateSessionId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ============================================================================
// JSON Utilities
// ============================================================================

/**
 * Safely stringify a value to JSON, returning null on failure.
 */
export function safeJsonStringify(value: unknown): string | null {
  try {
    return JSON.stringify(value);
  } catch {
    return null;
  }
}
