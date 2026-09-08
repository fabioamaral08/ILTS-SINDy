"""Modular test pipeline: data generation, method running, and analysis.

Built around two extension points:
- `pipeline.problems.Problem` — an ODE test case (see `pipeline/problems/`)
- `pipeline.methods.Method` — an identification algorithm (see `pipeline/methods/`)

Both are registered by name and looked up through small registries, so a new
problem or method is a single new file plus a one-line import, with nothing
else in the pipeline to touch.
"""
