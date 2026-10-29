# Configurations

The files under local/ demonstrate validated single-chain and parallel-tempering
inputs. Use `scripts/run.py CONFIG --validate-only` for single-chain or scan
inputs and `scripts/run_pt.py CONFIG --validate-only` for tempering inputs.

`mill.example.toml` contains placeholders for site configuration. Copy it to an
ignored `mill.local.toml` and supply your own account before using cluster tools.
Examples are not recommended production settings or authorization to submit jobs.
