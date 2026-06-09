# Full 166 Results Centralization

Structured-output constraint: LM Studio JSON schema `{code: string}`. This is general structured output, not XGrammar and not function-specific grammar.

Prompt-label correction: the Claude Haiku/Sonnet full-run directory says `prompt12`, but the user confirmed that was a typo. These Claude full-run results are treated as `prompt11` final results.

| Variant | Status | Model | Prompt | Constraint | Call | Exec | Official speed | Local speed | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Haiku + prompt 0 | done | claude-haiku-4-5 | prompt-0.txt | none | 71/166 (42.77%) | 55/166 (33.13%) | 0.46 | 1.03 | Generated and evaluated through Modal Anthropic path. |
| Sonnet + prompt 0 | done | claude-sonnet-4-6 | prompt-0.txt | none | 62/166 (37.35%) | 1/166 (0.60%) | 1.2 | 1.06 | Generated and evaluated through Modal Anthropic path. |
| Qwen local + prompt 0 | done | qwen/qwen3.6-35b-a3b via LM Studio | prompt-0.txt | none | 16/166 (9.64%) | 16/166 (9.64%) | n/a | n/a | Phase 3 efficiency stalled after 2/16 progress ticks; call/exec counts are from evaluator log and saved locally. |
| Qwen local + prompt 0 + general structured output | done | qwen/qwen3.6-35b-a3b via LM Studio | prompt-0.txt | LM Studio JSON schema: {code: string}; general, not function-specific; not XGrammar | 17/166 (10.24%) | 17/166 (10.24%) | n/a | n/a | Efficiency skipped after earlier Modal OOM; structured output only constrained envelope, not Triton semantics. |
| Qwen local + prompt 11 + general structured output | done | qwen/qwen3.6-35b-a3b via LM Studio | prompt-11-router.txt | LM Studio JSON schema: {code: string}; general, not function-specific; not XGrammar | 132/166 (79.52%) | 132/166 (79.52%) | 0.42 | 1.19 | Best of the five requested variants by correctness; prompt carries most of the improvement. |
| Existing Haiku + prompt 11 baseline | done | claude-haiku-4-5 | prompt-11-router.txt | none | 135/166 (81.33%) | 135/166 (81.33%) | 0.42 | 1.1 | Source directory says prompt12, but user confirmed this was a typo; treat as prompt11 final result. |
| Existing Sonnet + prompt 11 baseline | done | claude-sonnet-4-6 | prompt-11-router.txt | none | 159/166 (95.78%) | 159/166 (95.78%) | 0.4 | 1.23 | Source directory says prompt12, but user confirmed this was a typo; treat as prompt11 final result. |
| Existing Qwen local + prompt 11 baseline | done | qwen/qwen3.6-35b-a3b via LM Studio | prompt-11-router.txt | none | 139/166 (83.73%) | 139/166 (83.73%) | 0.4 | 1.13 | Existing full-166 prompt11 baseline found locally and included for context. |

## Individual Result Files

- Haiku + prompt 0: summary `outputs/full166_stats/summary_prompt0_haiku45.json`
- Sonnet + prompt 0: summary `outputs/full166_stats/summary_prompt0_sonnet46.json`
- Qwen local + prompt 0: summary `outputs/full166_stats/summary_qwen_prompt0.json`
- Qwen local + prompt 0 + general structured output: summary `outputs/full166_stats/summary_qwen_prompt0_structured_general.json`
- Qwen local + prompt 11 + general structured output: summary `outputs/full166_stats/summary_qwen_prompt11_structured_general.json`
- Existing Haiku + prompt 11 baseline: summary `outputs/full166_stats/summary_prompt11_haiku45.json`
- Existing Sonnet + prompt 11 baseline: summary `outputs/full166_stats/summary_prompt11_sonnet46.json`
- Existing Qwen local + prompt 11 baseline: summary `experiments/lmstudio_20260529-191522/latest-summary.json`

## Preflight / Eval Files

- Haiku + prompt 0: raw preflight n/a; repairs 0; summary `outputs/full166_stats/summary_prompt0_haiku45.json`
- Sonnet + prompt 0: raw preflight n/a; repairs 0; summary `outputs/full166_stats/summary_prompt0_sonnet46.json`
- Qwen local + prompt 0: raw preflight 163/166 valid; repairs 3 explicit-fail stubs in outputs/full166_stats/qwen_prompt0.eval.jsonl; summary `outputs/full166_stats/summary_qwen_prompt0.json`
- Qwen local + prompt 0 + general structured output: raw preflight 159/166 valid; repairs 7 explicit-fail stubs in outputs/full166_stats/qwen_prompt0_structured_general.eval.jsonl; summary `outputs/full166_stats/summary_qwen_prompt0_structured_general.json`
- Qwen local + prompt 11 + general structured output: raw preflight 164/166 valid; repairs 2 explicit-fail stubs in outputs/full166_stats/qwen_prompt11_structured_general.eval.jsonl; summary `outputs/full166_stats/summary_qwen_prompt11_structured_general.json`
- Existing Haiku + prompt 11 baseline: source `experiments/prompt12_claude_full_20260607-201510/haiku45/latest-summary.json`; summary `outputs/full166_stats/summary_prompt11_haiku45.json`
- Existing Sonnet + prompt 11 baseline: source `experiments/prompt12_claude_full_20260607-201510/sonnet46/latest-summary.json`; summary `outputs/full166_stats/summary_prompt11_sonnet46.json`
- Existing Qwen local + prompt 11 baseline: source `experiments/lmstudio_20260529-191522/lmstudio_qwen_qwen3.6-35b-a3b_simp_all_20260528-212826.jsonl`; summary `experiments/lmstudio_20260529-191522/latest-summary.json`
