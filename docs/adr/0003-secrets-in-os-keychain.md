# 3. Secrets live in the OS keychain, never in config files

- **Status:** Accepted
- **Date:** 2026-11-14

## Context

API keys in plain-text config files leak through backups, dotfile repositories and
screenshots. Yarvis also logs extensively, and logs are a classic place for secrets to
end up.

## Decision

- Secrets are stored in the OS keychain via `keyring`; an environment variable is the only
  fallback (for headless machines and CI). Insecure keyring backends (`fail`, `keyrings.alt`
  plaintext) are treated as unavailable.
- The config loader **rejects** any config file that appears to contain a credential.
- Every secret Yarvis resolves is registered with a process-wide redactor. All log handlers
  redact registered values and common credential formats, including in tracebacks.
- Rich and Typer "pretty" tracebacks, which print local variables, are disabled.
- Config, data and log files are created with mode 0600 inside 0700 directories.

## Consequences

Users run one extra command (`yarvis key set anthropic`). Linux users without a Secret
Service provider must use an environment variable, which `yarvis doctor` explains.
