# 13. MCP servers and Agent Skills, behind the same gate

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

Yarvis's own tools (timers, apps, music, mail, calendars, web search) cover the basics,
but nobody can add an ability without changing Yarvis. Two open standards fill that gap:

- the **Model Context Protocol** (MCP): a large ecosystem of servers, from files to GitHub,
  each offering tools with a description and a JSON Schema for their arguments;
- **Agent Skills** (agentskills.io): a folder with a `SKILL.md` (name, description,
  Markdown instructions) and optional reference files and scripts, loaded only when a task
  calls for it.

Both are also attack surface (design doc §5.2, T2): an MCP server is someone else's code,
its tool descriptions are text the model reads, and either can change after you trust it;
a skill is someone else's instructions. Other open-source assistants (OpenJarvis, for one)
install skills straight from any repository. Yarvis should be as extensible, and clearly
safer.

## Decision

### MCP servers (§5.4, now built)

1. **Opt-in, per tool.** A server runs only if it is in `[[mcp.servers]]`; only the tools
   listed under `[mcp.servers.tools]` are offered, each with the tier the user gives it
   (0 read only, 1 reversible, 2 asks first, 3 never). Unlisted tools are never offered.
2. **Pinned definitions.** `yarvis mcp approve <server>` starts the server and shows each
   listed tool exactly as the model would read it (description, arguments), then records
   a SHA-256 of name, description and schema. At every start the definitions are compared:
   a new or changed tool ("rug pull", tool poisoning) is left out, with a log line saying
   to approve it again. Unusable schemas (not an object, invalid, over 20 kB) are refused.
3. **Scrubbed processes.** A server Yarvis starts gets only the MCP SDK's harmless
   variables (PATH, HOME, ...), the ones listed under `env`, and its own `secrets`, which
   live in the OS keychain (`yarvis mcp secret <server> <NAME>`), never in the config. It
   runs in its own private folder; what it prints goes to a private `mcp-<name>.log`.
4. **Remote servers over HTTPS only**, with an optional bearer token from the keychain.
   OAuth is not supported yet.
5. **Outside content.** Every result is wrapped as untrusted data and taints the turn. A
   server's error message is outside content too: the model gets Yarvis's own summary plus
   the wrapped text (`UntrustedToolError`). Every MCP call counts as able to send data out
   (Yarvis cannot know where a server sends its input), so after reading outside content
   each one needs the user's "yes".
6. **Arguments are checked** against the approved JSON Schema (Draft 2020-12) before the
   gate; confirmations are read back from those checked arguments ("use notes' add note
   with text 'buy milk'"), built by code.
7. **Robust.** Each server is kept connected by its own task (the SDK's connections must be
   opened and closed by one task); a server that fails to start, hangs (30 s) or dies later
   is logged and left out, and Yarvis carries on. Servers start while the speech models
   load, so they add no startup time in the common case.

### Agent Skills

1. **Instructions only.** Yarvis reads `SKILL.md` and text reference files; scripts are
   never copied or run, and `allowed-tools` is ignored: nothing is pre-approved, every
   action goes through the gate.
2. **Added and pinned by the user.** `yarvis skills add <folder>` checks the skill against
   the spec (name rules, description length, front matter read with `yaml.safe_load`),
   shows it, and copies its text files into Yarvis's private folder with a fingerprint. A
   copy that changes later is left out until it is added again. Symbolic links are never
   followed, so a skill cannot point Yarvis at `~/.ssh`.
3. **Progressive disclosure.** The model sees each skill's name and description in the
   `use_skill` tool, loads the instructions only when a request calls for them, and reads
   reference files with `read_skill_file` (only the files the skill was added with).
4. **Outside content.** What a skill says is wrapped as untrusted data and taints the turn,
   like a web page; its tool description reminds the model that a skill is never a reason
   to skip a confirmation.

The prompt-injection suite (ADR 0009) now plants the same attack in an MCP tool result
and in a skill, and checks that it cannot act unasked.

## Consequences

- Anyone can extend Yarvis with existing MCP servers and skills, without touching its code.
- Adding an extension takes two deliberate steps (config, then approve/add), and every
  update of a tool's definition needs a new approval. That is the point.
- New dependencies: the official `mcp` SDK (MIT), `jsonschema` (MIT) and `PyYAML` (MIT).
- `yarvis doctor` reports missing secrets, unapproved tools and changed skills.
- Not yet: OAuth for remote servers, MCP resources and prompts, sampling and elicitation
  (Yarvis declines them), and per-server opt-outs of the "sends data out" rule.
