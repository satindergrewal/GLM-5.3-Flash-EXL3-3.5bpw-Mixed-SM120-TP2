# Evidence: real agentic client session (Spock over LAN)

Source: llama-webui conversation export, 2026-09-01 ~20:31-20:38 NZST
(08:31 UTC), against `lan:GLM-5.3-Flash-EXL3-3.5bpw` -> the 3.5bpw mixed
serve on port 8012. Raw export kept privately; metrics below only.

Model: GLM-5.3-Flash-EXL3-3.5bpw, thinking mode ON, MTP3, TP2 SM120.

## Decode rates observed by the client (predicted_n / predicted_ms)

| Turn | Prompt tokens | Generated tokens | Time | Rate |
|---|---|---|---|---|
| 1 (short first gen) | 2,607 | 154 | 12.9 s | 11.9 tok/s (first-gen outlier) |
| 2 | 2,784 | 750 | 14.7 s | 51.0 tok/s |
| 3, agentic turn 1 | 3,551 | 8,145 | 74.7 s | 109.0 tok/s |
| 3, agentic turn 2 | 17,282 | 4,431 | 47.5 s | 93.3 tok/s |
| aggregate | 20,833 | 12,576 | 122.1 s | 103.0 tok/s |

Client-side rates include network, SSE parsing, and UI overhead; the
engine-side rate for the same regime is 126-150 tok/s.

## What the session exercised

- Thinking mode: up to ~29,000 chars of reasoning content in a single turn.
- Tool calling: 2 x `web_search_exa` calls issued via OpenAI-format
  tool_calls, executed (1.2 s / 1.7 s), results fed back, both success.
  This is the glm45 tool parser path.
- Multi-turn agentic context growth: prompt 3,551 -> 17,282 tokens across
  turns within one request chain.
- First-gen outlier (turn 1) is a 154-token generation; short generations
  amortize worst. All longer generations sat at 51-109 tok/s.

## Boot receipt of the serve this ran against (boot 14, 2026-09-01)

```
GPU KV cache size: 1,985,915 tokens, Maximum concurrency for
  1,000,000 tokens per request: 1.99x
Capturing CUDA graphs (PIECEWISE): 4/4
Capturing CUDA graphs (FULL): 1/1
INFO: Application startup complete.   (08:02:35 UTC = 20:02 NZST)
```

- /v1/models: ["GLM-5.3-Flash-EXL3-3.5bpw"]
- Tool-call roundtrip test: tool_choice auto + get_weather ->
  finish_reason tool_calls, arguments {"city": "Auckland"}
- /health: 200
- Pack receipt (from materialization-receipt.json): bits
  mixed_k34_per_tensor, routed_choice_count 37,152 (18,576 K3 +
  18,576 K4), output_tensor_count 150,226, output_logical_bytes
  156,144,631,800 (145.4 GiB), complete: true
