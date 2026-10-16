# Physics conventions

This is the agreed project model, not a claim that another paper uses the
same parameterization. Simulation and analysis must share these definitions.

## Hamiltonian and ensemble

Real fields phi_i live on N = L^3 sites with periodic boundaries and six
nearest neighbors. Each undirected nearest-neighbor bond is counted once:

```text
H = -J sum_<ij> phi_i phi_j
    + sum_i [(r/2) phi_i^2 + (u/4) phi_i^4]
P(phi) proportional to exp(-H/T),  J = kB = 1, T > 0.
```

r < 0 and u > 0 give a local double well and stability. Working defaults
are r = -1, u = 1 (chosen here, not specified by the proposal). Keep them
fixed for a campaign. Tc must be measured for these couplings: an Ising
spin Tc or a Tc for an improved-action phi4 paper cannot be transplanted.
The temperature dependence is in exp(-H/T), not in independently retuned r.

For direct H evaluation, sum only the +x,+y,+z bonds. For L >= 3 there are
3N bonds. Restrict the first implementation to L >= 3 to avoid L = 1 self
bonds and L = 2 duplicate-neighbor/multigraph conventions.
Suggested flat index: i = x + L*(y + L*z), with x fastest.

## Local update identity

For phi_i = a -> b with neighbor sum S_i (six fields):

```text
delta_H = -J (b-a) S_i + (r/2)(b^2-a^2) + (u/4)(b^4-a^4).
```

This counts each changed bond once; there is no extra factor of 1/2.
Test against a separately written full-H calculation.

## Exact configuration checks

```text
Uniform field a: H/N = -3 J a^2 + (r/2) a^2 + (u/4) a^4.
Uniform minima: a^2 = (6J-r)/u, when 6J-r > 0.
Zero field: H = 0.
Global sign reversal: H(phi) = H(-phi).
```

The local on-site minima +/-sqrt(-r/u) differ from the uniform interacting
minima +/-sqrt((6J-r)/u). Use the latter for ordered starts. These are
zero-temperature identities, not estimates of Tc or a finite-T magnetization.

## Observables

Measure m = sum(phi_i)/N and e = H/N. Define ensemble averages after burn-in:

```text
B4 = <m^4> / <m^2>^2
U4 = 1 - B4/3  (only for explicitly labeled comparisons)
chi_abs = (N/T) [<m^2> - <|m|>^2]
chi_signed = (N/T) [<m^2> - <m>^2]
heat_capacity_per_site = (N/T^2) [<e^2> - <e>^2].
```

chi_abs is the proposal's finite-volume estimator suppressing sign switching;
it is not identical to the signed fluctuation-response susceptibility.
Never interchange their labels or fit data without recording which was used.
B4 approaches 1 for a sharply bimodal ordered distribution and 3 for a
centered Gaussian distribution. These are limiting checks, not exact values
at finite L and T. Compute ratios from averaged moments, not by averaging
m^4/m^2^2 sample by sample. Reject undefined or numerically unstable moments.

## Scaling expectations

For a continuous transition in the clean short-range Z2 model, the expected
class is 3D Ising. This is the proposal's hypothesis; it must be checked with
finite-size data. A [reference finite-size study](https://arxiv.org/abs/1004.4486)
reports nu = 0.63002(10), eta = 0.03627(10), and omega = 0.832(6).
Useful approximate comparison values are beta ~ 0.326 and gamma ~ 1.237,
obtained from beta/nu = (d-2+eta)/2 and gamma/nu = 2-eta for d = 3.
Do not treat the derived rounded values as new measurements or enforce them
in the fits. See [finite-size scaling](FINITE_SIZE_SCALING.md).
