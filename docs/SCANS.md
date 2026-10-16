# Size and temperature scans

A scan keeps one lattice and one random-number generator alive while changing T.
The coupling constants stay fixed. A random hot start cools from high to low T;
ordered positive or negative starts heat from low to high T.

## Configuration

Add a `[scan]` section to an ordinary run configuration:

```toml
[scan]
sizes = [4, 6, 8]
temperatures = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 1.0]
```

`model.temperature` must match the first temperature. The temperatures must be
strictly decreasing for `start = "random"` and strictly increasing for either
ordered start. The explicit list can use uneven spacing. All other parameters,
including J, r, u, seeds, and sweep counts, use the usual configuration fields.

`sizes` is optional; omitting it uses only model.L. If supplied, it must contain
unique integers from 3 to 512, and model.L must equal the first entry. Each size
starts a fresh lattice and uses seed = sampling.seed + its index in the size list.
The assigned seed is recorded in metadata. Reordering sizes changes those seeds.
Sizes run sequentially on the local machine, with the same sweep counts at each
size and temperature. For a size scan at one temperature, use temperatures = [T].

See `configs/local/scan.example.toml` and `configs/local/size-scan.example.toml`. Its range and sweep counts are exploratory
choices, not an established transition bracket or sufficient equilibration time.
To check it without starting a simulation:

```powershell
python scripts/run.py configs/local/scan.example.toml --validate-only
```

When ready to run, build first and pass the executable explicitly:

```powershell
cmake --preset local-release
cmake --build --preset local-release
python scripts/run.py configs/local/scan.example.toml --binary build/local-release/phi4.exe
```

On Linux the executable is `build/local-release/phi4`. Each attempt needs a new
run_id. To compare heating with cooling, use separate configs and seeds, reverse
the ladder for heating, choose an ordered sign, and change model.temperature.

## What happens in the code

1. `scripts/run.py` checks the ladder and saves the configuration, source archive,
   executable checksum, and `temperatures.txt` in the new run directory.
2. Python launches a separate C++ process for each size, writing logs under L4,
   L6, etc. `main()` constructs and initializes the sampler once per size. Its optional final
   argument is the temperature-file path, which selects `run_temperature_scan()`.
3. The scan changes `sampler.temperature` and calls the existing `run_chain()`
   at each point. Every point repeats adaptation, fixed-step settling, and
   measurements. Sweep counters restart at each temperature.
4. The fields, RNG state, and final step size carry into the next point.
   The step size can adapt again there before it is frozen for measurement.
5. Python checks every temperature's output and records all file checksums.
   A failure anywhere leaves the whole scan marked FAIL and retains partial data.

All new scans, including scans of one size, use this layout:

```text
data/raw/campaign/run-id/
    metadata.json
    config.resolved.toml
    source.zip
    temperatures.txt
    L4/
        stdout.log
        stderr.log
        temperature_0/
        temperature_1/
    L6/
        stdout.log
        stderr.log
        temperature_0/
        temperature_1/
```

`temperature_0`, `temperature_1`, etc. follow the requested ladder order.
Metadata records L, N, seed, temperature, and the relative path for every point.
Previously generated single-size scans remain readable at their original paths;
existing raw data is never moved. Non-scan single-temperature runs retain their
original flat output layout.
Each folder contains measurements.csv, equilibration.csv, initial_field.txt,
and final_field.txt. The final field at one point matches the initial field at
the next. The fields are boundary snapshots, not full restart checkpoints:
they do not save the RNG state, and no resume command is implemented.

## Preliminary plots

Use the scan-specific plot script; the ordinary `scripts/analyze.py` loader is
for single-temperature runs. Plotting does not start simulations.

```powershell
python scripts/plot_scan.py data/raw/temperature-pilot/L4-cooling-seed5409101-attempt01 --output results/figures/scan-attempt01 --block-size 100
```

One run directory produces one curve per size. Supply multiple scan directories
to overlay additional curves. The title gives the start; curve labels give L. Couplings must match.
The output includes temperature_scan.png/pdf, histories.png/pdf, summary.csv,
and plot_info.json. Existing plot directories are not overwritten.

The six panels show energy per site, mean absolute magnetization, mean signed
magnetization, normalized binder = (3 - B4)/2, susceptibility using |m|,
and heat capacity per site. The y-axis labels are <e>, <|m|>, <m>, binder,
<chi>, and c, respectively. The chi panel still uses the absolute-magnetization
susceptibility defined in PHYSICS.md. Binder approaches 1 in the ideal ordered
limit and 0 in the centered Gaussian limit; values are not clipped.
Error bars use the existing contiguous-block jackknife independently at each
T, requiring at least 20 complete blocks. Block size is in saved samples, not
sweeps. An incomplete final block is excluded from the estimates and reported
in the summary; histories show all production samples.

Check drift, compare heating and cooling and independent seeds, and vary block
size before interpreting the errors. A chosen block size is not an automatic
check of autocorrelation or equilibration. Temperatures in a scan share a
history; these plots do not fit Tc or exponents or treat the points as independent.
