# Local development and portability

## Windows route

Observed October 14, 2026: GCC 13.2.0, CMake 3.29.2, Ninja 1.12.0 under
`C:/msys64/mingw64/bin`; Python 3.13.1 and Git are available. Use these
tools first. No new compiler installation is currently indicated.
Availability/version checks are not evidence that a full simulator builds.

From PowerShell in the repository:

```powershell
cmake --preset local-debug
```

Rebuild after editing C++; the launcher no longer checks compiled-source
fingerprints. The project creates the sampler and five validation suites. Use `cmake --build --preset local-debug`, then
`ctest --test-dir build/local-debug --output-on-failure` after real tests
are registered. Empty CTest output is not validation.

If a new terminal cannot find the tools, check PATH for the installed
MinGW64 bin directory. Do not mix MSYS and MinGW compilers/libraries.
Keep Windows executables and Linux cluster executables in separate build
directories. CMake presets are portable; machine overrides belong in
ignored `CMakeUserPresets.json`.

## Python analysis environment (later)

Create a project virtual environment, without changing system Python:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip --version
```

Python's [venv documentation](https://docs.python.org/3/library/venv.html)
describes the environment layout. Calling its interpreter directly avoids
needing to change PowerShell activation policy.
Planned packages: NumPy, SciPy, Matplotlib, pytest, and Ruff. Select/install
and record compatible versions during implementation; no package install
or reproducibility lock has been performed in the scaffold stage.
Analysis should accept paths and run without notebook state. Notebooks
can explain results but must call the same tested analysis functions.

## Mill and build policy

Require CMake 3.21+ and a C++17 compiler. Mill modules must be checked live;
if CMake is missing use an appropriate module or environment installation
rather than adding a second hand-maintained build system.
Linux may use `cmake -S . -B build/mill-release -DCMAKE_BUILD_TYPE=Release`
with its available generator. Ninja is only required for the local presets.
Do not use `-ffast-math` initially. Keep release flags and compiler versions
in run metadata. Windows and Linux must agree statistically and pass the
same energy/observable tests; cross-platform RNG bitwise parity is not assumed.

Future C++ target warnings: GCC/Clang `-Wall -Wextra -Wpedantic`; MSVC `/W4`.
Debug safety checks and Linux sanitizers are separate from production
benchmarks. Tests should be small enough for local iteration. Add CI only
after targets exist so a green workflow actually means something.
