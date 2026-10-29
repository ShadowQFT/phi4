# Local development and portability

## Windows route

Requires a C++17 compiler, CMake 3.21+, Ninja, Python 3.11+, and Git.
The Windows setup has been checked with MinGW GCC 13.2.0 and Python 3.13.1.
Keep compiler installation paths in the local environment rather than in source.

From PowerShell in the repository:

```powershell
cmake --preset local-debug
```

Rebuild after editing C++; the launcher does not check compiled-source
fingerprints. Build and run the registered validation suites:

```powershell
cmake --build --preset local-debug
ctest --test-dir build/local-debug --output-on-failure
```

If a new terminal cannot find the tools, check PATH for the installed
MinGW64 bin directory. Do not mix MSYS and MinGW compilers/libraries.
Keep Windows executables and Linux cluster executables in separate build
directories. CMake presets are portable; machine overrides belong in
ignored `CMakeUserPresets.json`.

## macOS

Requires a C++17 compiler (Apple Clang is suitable), CMake 3.21+, Ninja, and
Python 3.11 or newer. The dependency versions were tested with Python 3.13 on
Windows; macOS has not been tested here. Use Python 3.13 for the same Python
version as that environment. If the compiler is missing, install Apple's
command-line tools with `xcode-select --install`.

From the repository directory, create a Python environment, install the analysis
dependencies, and configure CMake to use that environment's interpreter:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
cmake --preset local-release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset local-release
ctest --test-dir build/local-release --output-on-failure
```

The executable is build/local-release/phi4, without the Windows .exe suffix.
The class-demo spec contains the L=4, 8, 16 cooling scan used for the first plots:

```sh
python scripts/run.py configs/local/class-demo.toml --validate-only
python scripts/run.py configs/local/class-demo.toml --binary build/local-release/phi4
python scripts/plot_scan.py data/raw/testing/first-run --output results/figures/first-run --block-size 100
```

The second command runs the simulation. A retry needs a new run_id, and plotting
needs a new output directory if one already exists. Windows executables and
build caches are not portable; compile a fresh executable on the Mac. Library
random distributions can differ across toolchains, so equal seeds need not
produce identical Windows and macOS trajectories.

## Python environment on Windows

Use a project virtual environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r analysis/requirements.txt
```

Calling its interpreter directly avoids changing PowerShell activation policy.
NumPy and Matplotlib are used for analysis; the launcher itself uses Python's
standard library. Generated data, figures, and build directories are ignored by
Git. Personal specs ending in .local.toml are also ignored.

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
