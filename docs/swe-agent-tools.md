# SWE-agent Tool Landscape

This document compares the tool approaches used by SWE-agent variants.

## SWE-agent ReAct Tool Definitions

From `princeton-nlp/SWE-agent` ACI:

| Tool | Description |
|------|-------------|
| `str_replace_editor` | Custom file editor with 5 sub-commands: `view`, `create`, `str_replace`, `insert`, `undo_edit` |
| `bash` | Execute bash commands in the repo environment |
| `submit` | Submit current repo state as solution |

**`str_replace_editor` signature:**
```
str_replace_editor <command> <path> [<file_text>] [<view_range>] [<old_str>] [<new_str>] [<insert_line>]
```

## Mini-SWE-Agent Approach

From `lmareina/mini-swe-agent` (~100 lines Python):
- Single tool: bash command execution only
- Multi-turn agent loop with full conversation history
- Achieves ~74% resolution on SWE-bench Verified

## Comparison Table

| Approach | Tools | Turns | SWE-bench Verified | Complexity |
|----------|-------|-------|-------------------|------------|
| Zero-Tool (default) | None | 1 | ~10-15% | Low |
| Bash Agent (mini-swe-agent) | bash only | 50-100 | ~74% | Medium |
| SWE-agent ReAct | str_replace_editor + bash | 50-200 | ~47% | High |

## Configuration

```yaml
swebench:
  agent_mode: "bash"        # "zero-shot" (default) or "bash"
  max_iterations: 50        # max turns for bash agent
```

## Implementation Notes

The bash agent loop in this runner follows the mini-swe-agent pattern:
1. System prompt with repository context and issue description
2. Model receives bash tool definition
3. Model outputs bash commands via tool calls
4. Commands are executed (simulated for now, Docker-based in production)
5. Tool results are appended to conversation history
6. Loop continues until model outputs a diff patch or max_iterations reached
