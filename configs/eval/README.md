# MedARC Eval TOML Configs

These configs use upstream `verifiers` TOML semantics. Repeated `env_id` entries
and `[[ablation]]` sweeps intentionally keep the upstream environment id stable;
`medarc-eval bench` writes deterministic variant directories for differing
`env_args` and `sampling_args`.

Per-environment `[tool.verifiers.eval]` defaults are read from editable installs
where the environment `pyproject.toml` is discoverable next to the module. Wheel
installs may ignore those defaults unless the package includes `pyproject.toml`,
so production suite configs keep explicit `num_examples` and
`rollouts_per_example` values.
