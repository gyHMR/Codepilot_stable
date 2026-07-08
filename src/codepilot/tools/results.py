from __future__ import annotations

"""
工具结果规范化、脱敏与信任评估模块。

本模块负责对工具执行后的原始结果进行安全和质量处理，包括:
  1. 规范化（normalize）    — 统一结果格式、补充元数据、计算耗时
  2. 脱敏（sanitize）        — 检测并替换敏感信息（密钥、Token、PII）
  3. 提示注入检测              — 扫描结果中是否包含恶意指令
  4. 输出信任评估             — 对结果的可信度进行分级
  5. 大输出标记               — 识别超大输出并建议归档

设计原则:
  - 所有规则都是确定性的（正则匹配），不依赖 LLM 判断
  - 脱敏结果可审计（在 metadata 中记录 findings）
  - 信任评估基于来源（文件系统=可信，MCP=不可信）而非内容
"""

import re
import time
from dataclasses import dataclass
from typing import Pattern

from codepilot.protocols import TextContent
from codepilot.protocols.tools import ToolResultStatus, ensure_tool_result_status

from .contracts import ToolMetadata, ToolResult


@dataclass(frozen=True)
class _RedactionRule:
    """
    脱敏规则定义（不可变）。

    每条规则包含一个正则模式和一个替换文本。
    规则名称用于在 metadata 中记录检测结果。
    """

    name: str              # 规则名称（如 "private_key", "openai_key"）
    pattern: Pattern[str]  # 正则模式（已编译的 Pattern）
    replacement: str       # 替换文本（如 "[REDACTED_SECRET]"）


# ── 敏感信息脱敏规则 ─────────────────────────────────────────────────
# 编译器脱敏规则确保不删除——而是替换为明确的标记文本。
# 这保证结果的语义结构不变，同时防止敏感数据泄露。

# 密钥/Token 脱敏规则。
# 每条规则对应一类常见的敏感凭证格式。
_SECRET_RULES = (
    # PEM 格式私钥（包含 "PRIVATE KEY" 标记的完整块）
    _RedactionRule(
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_SECRET]",
    ),
    # 变量赋值中的敏感值: api_key=xxx, token=xxx, password=xxx 等
    _RedactionRule(
        "secret_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|token|secret|password|credential|cookie)\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=[REDACTED_SECRET]",
    ),
    # OpenAI API Key: sk- 前缀 + 20+ 字符
    _RedactionRule("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_SECRET]"),
    # GitHub Token: ghp_ 或 github_pat_ 前缀 + 20+ 字符
    _RedactionRule(
        "github_token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_SECRET]",
    ),
    # AWS Access Key: AKIA 前缀 + 16 位大写十六进制字符
    _RedactionRule("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_SECRET]"),
)

# 个人身份信息（PII）脱敏规则
_PII_RULES = (
    # 电子邮件地址
    _RedactionRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
)

# ── 提示注入检测规则 ─────────────────────────────────────────────────
# 检测工具输出中是否包含恶意指令模式。
# 这些模式是"提示注入"攻击的常见信号——外部内容试图覆盖模型的系统指令。

_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    # "忽略之前的所有指令" — 经典提示注入模式
    (
        "ignore_previous_instructions",
        re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    ),
    # "忽略之前的所有指令" 的另一种写法
    (
        "disregard_previous_instructions",
        re.compile(r"\bdisregard\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE),
    ),
    # "暴露系统提示词" — 试图获取 system prompt
    (
        "reveal_system_prompt",
        re.compile(r"\b(?:system prompt|developer message)\b", re.IGNORECASE),
    ),
    # 危险命令注入 — 试图执行破坏性操作
    (
        "dangerous_command_instruction",
        re.compile(r"\b(?:run\s+delete|execute\s+rm|delete_database)\b", re.IGNORECASE),
    ),
)

# 大输出阈值: 超过此字符数的工具输出标记为"大输出"
_LARGE_OUTPUT_CHARS = 30_000


@dataclass(frozen=True)
class ToolResultPolicy:
    """
    工具结果处理策略。

    定义了结果规范化的一套规则: 格式统一、脱敏、大输出标记。

    属性:
        large_output_chars: 大输出阈值（字符数），默认 30,000。
            超过此值的输出会标记 artifact_recommended=True，
            提示上层（ContextGovernor）将完整内容写入 artifact 文件。
    """

    large_output_chars: int = _LARGE_OUTPUT_CHARS

    def normalize(
        self,
        result: ToolResult,
        *,
        tool_call_id: str,
        tool_name: str,
        metadata: ToolMetadata | None,
        approval_id: str | None = None,
        permission_decision: dict[str, object] | None = None,
        started_at: float | None = None,
    ) -> ToolResult:
        """
        规范化工具执行结果。

        这是工具执行的最后一步，确保每个返回给 Agent 的结果都有统一格式。

        处理流程:
            1. 确保 result 是 ToolResult 类型（非 ToolResult 值自动包装）
            2. 绑定 tool_call_id 和 tool_name
            3. 确定有效状态（status 与 is_error 一致性修正）
            4. 注入审批信息（approval_id 和 approved 标志）
            5. 记录权限决策元数据
            6. 计算执行耗时（duration_ms）
            7. 处理空输出（补充 "(no output)" 文本）
            8. 执行脱敏和提示注入检测（_sanitize）
            9. 标记大输出（_mark_large_output）

        参数:
            result: 工具执行的原始结果。
            tool_call_id: 工具调用 ID。
            tool_name: 工具名称。
            metadata: 工具的 runtime metadata（用于信任评估）。
            approval_id: 审批 ID（如果有审批流程）。
            permission_decision: 权限决策记录。
            started_at: 执行开始时间（time.monotonic()），用于计算耗时。

        返回:
            规范化后的 ToolResult。
        """
        # 步骤 1: 确保类型正确
        if not isinstance(result, ToolResult):
            result = ToolResult(
                content=[TextContent(text=str(result))],
                status="success",
            )
        # 步骤 2-6: 注入标准元数据
        result.tool_call_id = tool_call_id
        result.tool_name = tool_name
        result.status = _effective_status(result)
        result.is_error = result.status != "success"
        if approval_id is not None:
            result.approved = True
            result.approval_id = approval_id
        if permission_decision is not None:
            result.metadata.setdefault("permission_decision", dict(permission_decision))
        if started_at is not None:
            # 计算实际执行耗时（毫秒）
            result.metadata.setdefault("duration_ms", int((time.monotonic() - started_at) * 1000))
        # 步骤 7: 空输出处理
        if not result.content:
            result.content = [TextContent(text="(no output)")]
            result.metadata.setdefault("output_quality", {"empty": True})
        # 步骤 8: 脱敏 + 注入检测
        self._sanitize(result, metadata=metadata)
        # 步骤 9: 大输出标记
        self._mark_large_output(result)
        return result

    def _sanitize(self, result: ToolResult, *, metadata: ToolMetadata | None) -> None:
        """
        对结果内容进行安全处理: 脱敏 + 提示注入检测。

        处理流程:
            1. 遍历所有 TextContent 块
            2. 对每个文本块执行脱敏（密钥、PII）
            3. 对每个文本块执行提示注入检测
            4. 汇总 findings 和注入检测结果
            5. 评估输出信任级别
            6. 将结果写入 metadata["result_guard"]

        注意: read 工具的结果会跳过密钥赋值模式的脱敏
              （因为源代码中可能包含类似 "api_key = xxx" 的合法内容），
              但仍会进行 PEM 密钥等其他类型的脱敏。
        """
        redacted = False
        findings: list[str] = []
        prompt_injection_suspected = False
        for block in result.content:
            if not isinstance(block, TextContent):
                continue
            # 执行脱敏: 根据工具类型选择规则集
            text, changed, block_findings = _redact_text(
                block.text,
                preserve_workspace_source=_preserve_workspace_source(metadata),
            )
            # 执行提示注入检测
            prompt_findings = _prompt_injection_findings(text)
            block.text = text
            redacted = redacted or changed
            prompt_injection_suspected = prompt_injection_suspected or bool(prompt_findings)
            findings.extend(block_findings)
            findings.extend(prompt_findings)
        # 去重
        findings = _unique(findings)
        # 评估输出信任级别
        output_trust = _output_trust(metadata, prompt_injection_suspected=prompt_injection_suspected)
        # 写入防护报告
        result.metadata["result_guard"] = {
            "redacted": redacted,
            "findings": findings,
            "prompt_injection_suspected": prompt_injection_suspected,
            "output_trust": output_trust,
        }
        result.metadata["output_trust"] = output_trust

    def _mark_large_output(self, result: ToolResult) -> None:
        """
        检测并标记超大工具输出。

        如果输出总字符数超过 large_output_chars 阈值，在 metadata 中
        写入 tool_artifact 记录，提示上层（ContextGovernor）将完整内容
        归档到文件，只在 prompt 中保留摘要。
        """
        total = sum(len(block.text) for block in result.content if isinstance(block, TextContent))
        if total <= self.large_output_chars:
            result.metadata.setdefault("output_quality", {}).setdefault("truncated", False)
            return
        result.metadata["tool_artifact"] = {
            "type": "tool_artifact",
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "original_chars": total,
            "preview_chars": self.large_output_chars,
            "summary": _summarize_blocks(result),
        }
        result.metadata.setdefault("output_quality", {})["artifact_recommended"] = True


def _effective_status(result: ToolResult) -> ToolResultStatus:
    """
    确定工具结果的实际状态。

    如果 is_error 为 True 但 status 是 "success" → 修正为 "error"。
    确保 status 与 is_error 标志保持一致。
    """
    status = ensure_tool_result_status(result.status)
    if result.is_error and status == "success":
        return "error"
    return status


def _redact_text(
    text: str,
    *,
    preserve_workspace_source: bool,
) -> tuple[str, bool, list[str]]:
    """
    对文本执行敏感信息脱敏。

    参数:
        text: 原始文本。
        preserve_workspace_source: True 时使用工作区源码规则集
            （只脱敏 PEM 密钥等绝对敏感内容，不脱敏变量赋值模式，
            因为源码可能包含合法的 api_key=xxx 赋值）。
            False 时使用完整规则集（密钥 + PII）。

    返回:
        (脱敏后文本, 是否发生了脱敏, 检测到的规则名称列表)
    """
    # 根据是否为源码选择规则集
    rules = _workspace_source_redaction_rules() if preserve_workspace_source else (
        *_SECRET_RULES,
        *_PII_RULES,
    )
    redacted = False
    findings: list[str] = []
    guarded = text
    for rule in rules:
        guarded, count = rule.pattern.subn(rule.replacement, guarded)
        if count:
            redacted = True
            findings.append(rule.name)
    return guarded, redacted, findings


def _workspace_source_redaction_rules() -> tuple[_RedactionRule, ...]:
    """
    返回工作区源码专用的脱敏规则集。

    排除 secret_assignment 规则，因为源码中的赋值语句
    （如 `api_key = "xxx"`) 可能是合法的代码片段，不应该被脱敏。
    但仍保留 PEM 密钥、API Key 令牌等其他规则的检测。
    """
    return tuple(rule for rule in _SECRET_RULES if rule.name != "secret_assignment")


def _preserve_workspace_source(metadata: ToolMetadata | None) -> bool:
    """
    判断是否应该使用工作区源码脱敏规则。

    只有 read 工具 + filesystem 分类时才使用弱规则集，
    因为此时工具正在读取项目源码文件，源码中的赋值不应被脱敏。
    """
    return metadata is not None and metadata.name == "read" and metadata.category == "filesystem"


def _prompt_injection_findings(text: str) -> list[str]:
    """
    扫描文本中是否存在提示注入攻击模式。

    返回匹配到的规则名称列表（空列表 = 未检测到）。
    """
    return [name for name, pattern in _PROMPT_INJECTION_PATTERNS if pattern.search(text)]


def _output_trust(
    metadata: ToolMetadata | None,
    *,
    prompt_injection_suspected: bool,
) -> str:
    """
    评估工具输出的信任级别。

    评估优先级:
        1. 提示注入检测到 → "untrusted"（不可信）
        2. 工具 metadata 显式配置了 output_trust → 使用配置值
        3. MCP 工具或扩展工具 → "untrusted"（外部来源不可信）
        4. 有网络访问的工具 → "untrusted"
        5. 默认 → "local"（本地工具结果可信）

    返回: "trusted" | "local" | "untrusted" | "sanitized"
    """
    if prompt_injection_suspected:
        return "untrusted"
    configured = metadata.extra.get("output_trust") if metadata is not None else None
    if configured in {"trusted", "local", "untrusted", "sanitized"}:
        return str(configured)
    # MCP 和扩展工具的输出默认不可信
    if metadata is not None and (
        metadata.category in {"mcp", "extension"} or metadata.network_access
    ):
        return "untrusted"
    # 本地内置工具的输出默认可信
    return "local"


def _summarize_blocks(result: ToolResult) -> str:
    """
    从 ToolResult 中提取摘要文本。

    取所有 TextContent 块拼接后的第一行非空内容，
    截断到 300 字符。如果全部为空，返回默认摘要。
    """
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:300] or f"{result.tool_name} produced {len(text)} characters"


def _unique(values: list[str]) -> list[str]:
    """
    字符串列表去重，保持首次出现的顺序。
    """
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


__all__ = ["ToolResultPolicy"]
