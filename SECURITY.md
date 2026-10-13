# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository. Do not open a public issue.

## How shiftlock handles your data

- **API keys** are read only from `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. They are never
  accepted as command-line arguments or written to disk. Logs and errors redact anything that
  looks like a key.
- **Prompts and outputs** go only to the providers you configure, using your own keys. Nothing
  is sent anywhere else: there is no telemetry and no update check.
- **Local files** under `.shiftlock/` (suites, raw results, reports) contain your prompts. They
  are created owner-only (`0700`/`0600` on POSIX systems), and the directory carries its own
  `.gitignore` so they are not committed by accident. On Windows, rely on your user profile's
  permissions.
- **Import** flags cases that look like they contain emails or phone numbers and asks before
  keeping them. This is a simple pattern check, not a guarantee; review your data before
  importing it.
