# Validation gates

These are acceptance criteria; actual local/pending evidence is in STATUS.md.
Physics-bearing changes require independent expected values or identities.

| Gate | Check | Evidence required |
| --- | --- | --- |
| V1 topology | Periodic wrapping, six neighbors, bond count, L=3/4 fixtures | Explicit coordinate expectations independent of production helpers |
| V2 Hamiltonian | Zero/uniform fields, sign symmetry, translations | Analytic H/N and direct bond enumeration |
| V3 local changes | Random fields/site/proposal, boundary sites | delta H against separate full H; mixed absolute/relative tolerance |
| V4 transition | Uphill/downhill cases, T factors, symmetric proposals | Controlled RNG inputs and independently calculated acceptance |
| V5 sampler | J=0 on-site distribution | Moments from numerical quadrature with autocorrelation-aware uncertainty |
| V6 observables | Fixed synthetic distributions | Known moments, B4, N/T factors; constant/zero edge cases |
| V7 statistics | Correlated synthetic series, nonlinear ratios | Known variance inflation and resampling consistency |
| V8 workflow | Tiny Windows and Mill run | Config, metadata, series, exit status, seed replay on same binary |
| V9 equilibrium | Ordered/disordered and split histories | Stable e, |m|, m^2 with justified burn-in and block errors |
| V10 scaling | Tc, windows, Lmin, correlated derived exponents | Refit sensitivity, residuals, bootstrap intervals, clear exclusions |

V1-V4 precede useful simulation runs. V5-V8 precede production. V9-V10
determine whether a Friday result is provisional or suitable for a stronger
claim. Local smoke tests verify operation; they do not establish criticality.

Use double precision. Begin with absolute/relative tolerances around 1e-10
for small direct-energy tests and explain changes as sizes and sums grow.
Avoid fragile stochastic CI tests: keep deterministic identities in the
routine suite and long distribution checks in an explicitly invoked suite.

For J=0, integrate exp[-(r phi^2/2 + u phi^4/4)/T] over the real line,
using controlled numerical tails. Compare several even moments. This
oracle cannot reuse Metropolis output to set its expected answer.

Analytic expectations near criticality are not a license to calibrate test
tolerances around a desired exponent. Track failures and incomplete gates
in `STATUS.md`; never replace a failed test with a looser undocumented check.
