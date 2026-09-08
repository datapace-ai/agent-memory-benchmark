# Local track, kept for reference

The first phase of this benchmark ran on a local model through Ollama
(qwen3:14b with reasoning forced off, nomic-embed-text for embeddings). The
published results do not use it: every reported number comes from free models
on OpenRouter, configured in `configs/models.yaml` and `configs/tracks.yaml`.

These files are kept so that anyone who prefers to run offline can:

- `models.ollama.yaml`: the model configuration for the local track.
- `create_nothink_model.sh` and `Modelfile.qwen3-nothink`: build the
  `qwen3-nothink:14b` variant for libraries that reach Ollama through its
  OpenAI-compatible endpoint, which has no reasoning flag.
- `letta_server.sh`: a self-hosted Letta server pointed at Ollama.

To use it: install Ollama, pull the two models, then run the harness with
`--models-config local-track/models.ollama.yaml`. Expect about 6.5 hours per
seed for the two baselines on an Apple M-series laptop, as measured in
`METHODS.md`.
