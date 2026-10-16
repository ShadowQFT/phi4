# Reproducibility checklist

Before a run: freeze source identity, resolved input, seed scheme,
measurement definitions, environment/compiler, and resource budget.
Record a clean commit when available; a dirty tree needs a content snapshot
hash and archived diff. A commit ID alone is insufficient for a dirty build.

After a run: preserve time series and failure records, check expected counts
and status, hash files, mirror required scratch data, and record the pull.
Do not erase raw data because a derived summary exists.

For analysis: keep ordered raw inputs, exclusions, scripts/commands,
statistical choices, environment, fit ranges, confidence level, and provenance.
Repeated sampling and bootstrap seeds are distinct recorded identities.
Derived files and plots can be regenerated; deleting them must never remove
their raw inputs. Preserve presentation artifacts separately from working plots.

Before presentation: regenerate from saved inputs, inspect the actual plots,
check observable labels and units, reconcile tables with fit records, and
include a short limitations note. Separate literature values, synthetic
fixtures, pilot results, and production estimates visually and in captions.
