"""crucible-bench — benchmark a served model over an OpenAI-compatible API.

Deliberately separate from `crucible`. Compression produces weights;
benchmarking measures an endpoint. They share no imports and no dependency
tree — this package needs neither torch nor transformers, so it installs on a
laptop that only ever talks to a remote server.

The interface between the two is files, not imports: crucible writes a model
and its `compression_metadata.json`, and crucible-bench writes result JSON
that `compare` diffs.
"""
