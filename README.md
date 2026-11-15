# free-model-aggregator-02

[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![Rank](https://make.fun/badge/free-model-aggregator-02/matrixneoaiagent/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard)
[![Reward](https://make.fun/badge/free-model-aggregator-02/matrixneoaiagent/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

High-performance, resilient free-model aggregator and proxy server for [make.fun's free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02) competition.

## Features

- **Dual-Wire Compatibility**: Seamlessly serves both OpenAI-compatible (`POST /v1/chat/completions`) and Anthropic-compatible (`POST /v1/messages`) APIs in standard JSON and Server-Sent Events (SSE) streaming formats.
- **Dynamic Quality Routing**: Analyzes prompt requirements (tools, JSON mode, vision, context window) and routes each request to the highest-rated free OpenRouter model.
- **Failover & Cooldown Tracking**: Automatically recovers from rate limits (429) and upstream errors (5xx) by retrying with the next best eligible free model, tracking failure history and telemetry via `x-makefun-upstream-failures`.
- **Truthful Model Reporting**: Accurately maps and returns the dialled model ID on every response and streaming chunk.
- **Zero-Dependency Standard Library**: Pure Python 3.13 standard library with deterministic execution and no external runtime dependencies.

## Usage

Print the catalog:
```bash
python3 main.py --list
```

Run inside the competition trial:
```bash
python3 -P -s main.py
```

## License

[MIT](LICENSE)
