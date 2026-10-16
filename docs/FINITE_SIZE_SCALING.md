# Finite-size scaling analysis

Temperature Tc is nonuniversal; exponents are the comparison target.
Use identical fixed couplings and definitions at every size.

## Find Tc

1. Use small pilot lattices to bracket the ordered/disordered region.
2. Refine an overlapping T grid for L=8,12,16,24 after pilot results.
3. Fit/interpolate B4(T,L) over narrow justified windows; do not extrapolate
   crossings outside measured coverage.
4. Find pair crossings with block-bootstrap intervals. Shared sizes create
   correlated crossings; do not call them independent observations.
5. Examine crossing drift. For pairs L,sL, a leading form is
   T_cross(L,sL) = Tc + A(s)*L^[-(omega + 1/nu)]. Amplitudes depend on size
   ratio; do not fit one A to mixed ratios (e.g. 8/12 and 8/16).

Four sizes support only limited correction fits. Report a finite-volume
crossing-based Tc estimate if drift cannot be resolved. Propagate the
plausible Tc range and clearly label remaining systematic uncertainty.
Do not fix the critical Binder value from another convention or geometry.

## Extract exponents

```text
s_L = |d B4 / dT| evaluated at Tc  ~ L^(1/nu)
<|m|>(Tc,L)                     ~ L^(-beta/nu)
chi_abs(Tc,L)                   ~ L^(gamma/nu)
```

Obtain slopes from local temperature fits rather than noisy finite
differences between only two measurements. Record the interval, polynomial
order, weights, and derivative uncertainty. Inspect fit curvature/residuals.
If temperature coverage cannot constrain a slope, withhold nu.

Weighted log-log fits give p_nu, p_m, p_chi; then nu=1/p_nu,
beta=-nu*p_m, gamma=nu*p_chi. Assess uncertainty with the joint resampling
pipeline from [statistics](STATISTICS.md). Positive finite measurements and
a suitable error model are required before taking logarithms. Native-space
weighted power-law fits are a cross-check when relative errors are large.

Repeat at varied Tc, Lmin, and local T windows. An optional form is
Q(L)=A*L^p*(1+b*L^-omega); do not fit A,p,b,omega freely to four points.
Any fixed omega is an external assumption and must be disclosed. With
three sizes left after removing the smallest, a two-parameter power law
has only one residual degree of freedom; good-looking lines are weak evidence.

## Physics checks and meeting language

Check 2 beta/nu + gamma/nu ~ 3 (hyperscaling) as a diagnostic; do not
enforce it to improve apparent agreement. The reference values are in
[physics](PHYSICS.md) and must remain comparisons, not fitting targets.
B4 and chi_abs are nonlinear estimates: fit uncertainty must use joint
moment resampling rather than a ratio of independently perturbed moments.

For every fit save sizes, Tc treatment, temperature windows, input hashes,
parameters, covariance/intervals, residuals, fit statistic, degrees of
freedom, and excluded points. If the Friday sample is insufficient,
present preliminary crossings and effective finite-size slopes with the
checks still missing; do not label them precise asymptotic exponents.
