# Real XGrammar Engine Investigation

## Verdict

The fastest defensible path is **Modal + SGLang with XGrammar EBNF constraints**.
LM Studio can keep serving the unconstrained baseline, but the current local API
does not expose a true XGrammar grammar parameter for arbitrary Python grammar
decoding. Our probe in `outputs/xgrammar_lmstudio_probe.json` showed:

- top-level `grammar` was accepted by HTTP but ignored;
- `response_format.type = "grammar"` was rejected;
- native `/api/v1/chat` rejected `grammar`.

That means the previous LM Studio grammar experiments were useful, but not true
XGrammar-constrained decoding.

## Why LM Studio Is Not Enough

XGrammar works by creating a token mask at every generation step and applying it
to logits before sampling. The XGrammar docs describe this as masking invalid
tokens to `-inf` so they cannot be sampled.

LM Studio's structured-output docs currently document JSON schema output. For
GGUF models, LM Studio says it uses llama.cpp grammar-based sampling APIs; for
MLX it uses Outlines. That is grammar constrained, but it is not the XGrammar
backend requested by the challenge.

Local Qwen in LM Studio is also GGUF:

```text
~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/
  Qwen3.6-35B-A3B-Q4_K_M.gguf
```

So the local app route is a good fallback for llama.cpp-style grammar sampling,
not for claiming "Use XGrammar".

## Viable Routes

| Route | Real XGrammar? | Uses our Qwen3.6 model? | Notes |
|---|---:|---:|---|
| Modal + SGLang | Yes | Yes, HF model | Recommended. SGLang docs say XGrammar is the default grammar backend and `ebnf` constrains output. |
| Modal + vLLM | Yes | Yes, HF model | Also viable. vLLM uses `structured_outputs: {grammar: ...}` and supports XGrammar/guidance backends. |
| Mac + HF Transformers + xgrammar | Yes | Only if HF weights fit | Good small-model proof; not realistic for Qwen3.6 35B on this Mac. |
| Mac + llama.cpp / LM Studio GGUF | No, not XGrammar | Yes, local GGUF | Useful fallback, but weaker for the course requirement. |

## Recommended Experiment

Use `grammars/triton_python_xgrammar.ebnf` as the general grammar and run four
conditions:

1. `prompt0`
2. `prompt11`
3. `prompt0 + XGrammar`
4. `prompt11 + XGrammar`

The new harness is `modal_app_xgrammar.py`. It launches SGLang on Modal, passes
the EBNF grammar through `extra_body={"ebnf": grammar}`, writes four local JSONL
files, and prints the `evaluate_only` commands.

```bash
XGRAMMAR_GPU=H100 python3 -m modal run modal_app_xgrammar.py::compare \
  --limit 20 \
  --operations add,sub,sqrt,rsqrt,tanh,relu_sqrt \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --tp-size 1 \
  --context-length 32768
```

If one H100 does not fit the model/context, retry with more GPU memory or tensor
parallelism:

```bash
XGRAMMAR_GPU=H100:2 python3 -m modal run modal_app_xgrammar.py::compare \
  --limit 20 \
  --operations add,sub,sqrt,rsqrt,tanh,relu_sqrt \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --tp-size 2 \
  --context-length 32768
```

For the final paper, the important phrasing is:

> We used SGLang's structured-output interface with the XGrammar backend to
> enforce a Triton/Python EBNF grammar during token sampling.

## Sources

- XGrammar constrained decoding: https://xgrammar.mlc.ai/docs/tutorials/constrained_decoding.html
- XGrammar workflow and HF integration: https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html
- SGLang structured outputs: https://docs.sglang.io/docs/advanced_features/structured_outputs
- SGLang sampling parameters: https://docs.sglang.io/docs/basic_usage/sampling_params
- vLLM structured outputs: https://docs.vllm.ai/en/v0.15.0/features/structured_outputs/
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6-35B-A3B-FP8 model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
- LM Studio structured output: https://lmstudio.ai/docs/developer/openai-compat/structured-output
