# Validation

CTest registers model, distribution, workflow, statistics, analysis, scan,
scaling, tempering, PT workflow, and PT statistics suites. Small operational
fixtures and synthetic analysis inputs are written under build/.

Run `ctest --test-dir build/local-debug --output-on-failure` after building.
The complete suite should pass on the source revision being checked. Local
uncommitted changes must be distinguished from that revision when reporting
results. These software checks do not establish equilibration or criticality.

See [validation criteria](../docs/VALIDATION.md) for the independent oracles.
