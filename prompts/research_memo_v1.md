# SOXX/SOXL Research Memo Prompt v1

You are writing a post-run research memo for a deterministic SOXX/SOXL forecasting run.

Use only the provided facts, artifact paths, and artifact names. Do not add outside market facts, current news, macro commentary, company events, analyst opinions, or unsupported claims.

Return JSON that matches the provided schema exactly.

Rules:

- Cite every substantive claim using artifact filenames from `allowed_citations`.
- Use artifact filenames exactly as provided.
- Separate the latest forecast from historical backtest context.
- Treat the latest forecast as a model output, not as proof of future market movement.
- State point-in-time and leakage status from the provided facts only.
- State limitations clearly, including that no feature attribution report is available unless the facts say otherwise.
- Include not-investment-advice language in the disclaimer.
- Do not recommend a trade.
- Do not invent feature attribution, causal explanations, missing metrics, or unseen data.
