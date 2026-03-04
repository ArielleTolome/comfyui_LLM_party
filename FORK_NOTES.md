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
- [ ] Fix #227: Ollama streaming returns empty string in ComfyUI output
- [ ] Fix #222: numpy 2.0 compatibility (`np.float_` → `np.float64`)
- [ ] Fix #224: macOS install issues

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
