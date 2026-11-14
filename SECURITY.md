# Security policy

Yarvis listens to a microphone and can act on your computer, so we take security
seriously. The security model is described in [docs/DESIGN.md §5](docs/DESIGN.md#5-security-model).

## Reporting a vulnerability

**Please do not open a public issue.** Report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on this repository.

Include what you found, how to reproduce it, and the impact you expect. You can expect:

- an acknowledgement within 72 hours;
- an assessment and, if confirmed, a fix plan within 14 days;
- credit in the release notes, if you want it.

## In scope

- Bypasses of the permission gate or confirmation flow (once implemented in Phase 2).
- Prompt injection that leads to an action the user did not confirm.
- Leaks of API keys, audio, transcripts or memories (to disk, logs, network or other users).
- Weaknesses in the satellite pairing or transport (Phase 4).
- Supply-chain issues in our build and release process.

## Out of scope

- The LLM saying something wrong or rude without taking an unconfirmed action.
- Attacks that require an attacker who already controls your user account.
- Vulnerabilities in third-party MCP servers you chose to install (report those upstream).

## Supported versions

Until v1.0, only the latest commit on `main` receives security fixes.
