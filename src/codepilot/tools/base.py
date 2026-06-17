from __future__ import annotations

"""Tool protocol facade.

This module exists as the stable import point for future first-class Tool
classes. During the compatibility phase, CodePilot still exposes AgentTool
objects to the LLM/core layer.
"""

from .types import AgentTool, AgentToolResult, ToolMetadata

__all__ = ["AgentTool", "AgentToolResult", "ToolMetadata"]
