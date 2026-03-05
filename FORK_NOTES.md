# Fork Notes — ArielleTolome/comfyui_LLM_party

This is an actively maintained fork of [heshengtao/comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party).

## Why This Fork?

The original repo's last commit was **Sep 8, 2025**. This fork aims to:
- Fix open bugs from the upstream issue tracker
- Add support for newer APIs (Grok-3, Kimi k1.5, GLM-4-Flash, MiniMax-Text-01)
- Improve English documentation
- Keep dependencies up to date (numpy 2.x, httpx 0.28+)
- Add starter workflows

## Improvements Over Upstream

### Bug Fixes
- [x] Fixed SyntaxWarning: invalid escape sequences in `llm.py` (Windows path strings)
- [x] Fix #227: Ollama streaming returns empty string in ComfyUI output
- [x] Fix #219: `expected string or bytes-like object, got 'NoneType'` with Gemini
- [x] Fix #187 + #204: Widget validation errors on old workflows (`is_enable_system_role`, `conversation_rounds`, `historical_record` coercion)
- [x] Fix #232: `show_text_party` shows stale stacked blocks when UI text carries list payload — render latest item only
- [x] Fix #230: GPT-5 chat/completions fails because node always sends `max_tokens` — remapped to `max_completion_tokens` for `gpt-5*` models
- [x] Fix #191: Janus Pro (`ds_chat`) ignores seed and temperature — `do_sample` now respects temperature; `seed` extracted from extra_parameters and applied via `torch.manual_seed()`
- [x] Fix #191 (all local models): `seed` kwarg in extra_parameters passed to HuggingFace `.generate()` caused "model_kwargs not used" errors — intercepted in all four local chat functions (`llm_chat`, `llama_chat`, `qwen_chat`, `ds_chat`)
- [x] Fix #190: `httpx<=0.27.2` pin removed — aisuite ≥0.1.12 supports httpx 0.28+; updated to `httpx>=0.27.0`
- [ ] Fix #222: numpy 2.0 compatibility — transitive dependency via spacy/cupy; not present in our codebase, requires upstream langchain-text-splitters fix
- [ ] Fix #224: macOS install issues (platform not applicable)

### New Features Planned
- [ ] Native Grok-3 / Grok-3-mini model presets
- [ ] Native Kimi k1.5 / moonshot presets  
- [ ] Native GLM-4-Flash presets
- [ ] MiniMax-Text-01 support
- [ ] Starter workflow pack (prompt generation, system prompt templates)
- [ ] English-first documentation overhaul

## Syncing With Upstream

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Contributing

PRs welcome. Focus areas:
1. Bug fixes for open upstream issues
2. API compatibility updates
3. New model provider presets
4. English translations of Chinese-only docs
