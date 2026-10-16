#include "phi4/sampler.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
using SweepCount = unsigned long long;

struct SweepCounts {
    SweepCount adaptation;
    SweepCount settling;
    SweepCount measurement;
    SweepCount interval;
};

void validate_counts(const SweepCounts &sweeps) {
    constexpr SweepCount maximum_sweeps = 1000000000ULL;
    if (sweeps.interval == 0 || sweeps.measurement == 0 || sweeps.settling == 0 ||
        sweeps.adaptation > maximum_sweeps || sweeps.settling > maximum_sweeps ||
        sweeps.measurement > maximum_sweeps) {
        throw std::invalid_argument("Invalid sweep counts");
    }
}

void write_measurement(std::ostream &output_file, const phi4::Sampler &sampler) {
    // Recompute H rather than accumulating local changes and numerical drift.
    const double energy =
        phi4::energy(sampler.lattice, sampler.model) / sampler.lattice.field.size();
    const double magnetization = sampler.magnetization();
    if (!std::isfinite(energy) || !std::isfinite(magnetization)) {
        throw std::runtime_error("Nonfinite measurement");
    }
    output_file << energy << ',' << magnetization;
}

// Burn-in samples are written for inspection, but never used as production samples.
void burn_in(phi4::Sampler &sampler, SweepCount count, bool do_adapt, std::ostream &output_file,
             SweepCount &total_sweeps) {
    double acceptance_sum = 0;
    for (SweepCount j = 1; j <= count; ++j) {
        const double used_step = sampler.step_size; 
        const double acceptance = sampler.sweep(); // gives acceptance rate for this sweep
        ++total_sweeps;
        acceptance_sum += acceptance;

        // Write current measurement and acceptance info to output file
        std::string current_sweep = std::to_string(total_sweeps);
        output_file << current_sweep << ',';
        write_measurement(output_file, sampler);
        output_file << ',' << acceptance << ',' << used_step << ',' << (do_adapt ? "adapt" : "settle")
                    << '\n';

        // Adapt the step size every 50 sweeps if in the adaptation phase.
        if (do_adapt && j % 50 == 0) {
            // mean acceptance over the last 50 sweeps
            const double mean_acceptance = acceptance_sum / 50;
            // Adjust by 10% toward 40-60% acceptance
            if (mean_acceptance < 0.4) {  // too low, decrease step size
                sampler.step_size *= 0.9;
            } else if (mean_acceptance > 0.6) {  // too high, increase step size
                sampler.step_size *= 1.1;
            }
            sampler.step_size = std::clamp(sampler.step_size, 1e-4, 100.0); 
            acceptance_sum = 0;
        }
    }
}

void run_chain(phi4::Sampler &sampler, const SweepCounts &sweeps,
               const std::string &output_directory) {
    // Open output files 
    std::ofstream equilibration_file(output_directory + "/equilibration.csv");
    std::ofstream measurement_file(output_directory + "/measurements.csv");
    equilibration_file.exceptions(std::ios::failbit | std::ios::badbit);
    measurement_file.exceptions(std::ios::failbit | std::ios::badbit);

    // Write output headers
    equilibration_file << std::setprecision(17)
                       << "sweep,energy_per_site,magnetization,acceptance,step_size,phase\n";
    measurement_file << std::setprecision(17) << "sample_index,sweep,energy_per_site,magnetization\n";

    SweepCount total_sweeps = 0; // total number of sweeps
    SweepCount measurement_count = 0; // total number of measurements

    // adaptation phase
    burn_in(sampler, sweeps.adaptation, true, equilibration_file, total_sweeps);

    // settling phase
    burn_in(sampler, sweeps.settling, false, equilibration_file, total_sweeps);

    // measurement phase
    for (SweepCount j = 1; j <= sweeps.measurement; ++j) {
        sampler.sweep();
        ++total_sweeps;
        if (j % sweeps.interval == 0) {
            measurement_file << measurement_count++ << ',' << total_sweeps << ',';
            write_measurement(measurement_file, sampler);
            measurement_file << '\n';
        }
    }

    // A close/write failure must reach the launcher as a nonzero exit.
    equilibration_file.close();
    measurement_file.close();
    std::cout << std::setprecision(17) << "{\"measurement_count\":" << measurement_count
              << ",\"final_step_size\":" << sampler.step_size << "}\n";
}
// Save the field at temperature boundaries so continuation can be inspected.
void write_field(const phi4::Sampler &sampler, const std::string &path) {
    std::ofstream output_file(path);
    output_file.exceptions(std::ios::failbit | std::ios::badbit);
    output_file << std::setprecision(17);
    for (double field : sampler.lattice.field) {
        output_file << field << '\n';
    }
    output_file.close();
}

void run_temperature_scan(phi4::Sampler &sampler, const SweepCounts &sweeps,
                          const std::string &output_directory,
                          const std::string &temperature_file_path) {
    // Read the whole ladder before running anything. Python sets its direction.
    std::ifstream temperature_file(temperature_file_path);
    if (!temperature_file) {
        throw std::invalid_argument("Cannot open temperature file");
    }

    std::vector<double> temperatures;
    double temperature;

    // validate temperatures
    while (temperature_file >> temperature) {
        if (!std::isfinite(temperature) || temperature <= 0) {
            throw std::invalid_argument("Scan temperatures must be finite and positive");
        }
        temperatures.push_back(temperature);
    }
    if (!temperature_file.eof() || temperatures.empty()) {
        throw std::invalid_argument("Require at least one valid scan temperature");
    }

    // Run the simulation for each temperature in the ladder
    for (std::size_t index = 0; index < temperatures.size(); ++index) {
        // set current temp
        sampler.temperature = temperatures[index];

        // create output directory for this temperature
        const std::string temperature_directory =
            output_directory + "/temperature_" + std::to_string(index);
        if (!std::filesystem::create_directory(temperature_directory)) {
            throw std::runtime_error("Temperature output directory already exists");
        }

        // write initial field to file
        write_field(sampler, temperature_directory + "/initial_field.txt");

        // run simulation
        run_chain(sampler, sweeps, temperature_directory);

        // write final field to file
        write_field(sampler, temperature_directory + "/final_field.txt");

    }
}
} // namespace

// Public entry point: scripts/run.py validates TOML before calling this program.
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            std::cout << "C++17; compiler=" << PHI4_COMPILER << "; mode=" << PHI4_BUILD_TYPE
                      << '\n';
            return 0;
        }
        if (argc != 14 && argc != 15) {
            throw std::invalid_argument("Use scripts/run.py CONFIG --binary PATH");
        }

        // cmd arguments -> variables
        const int L = std::stoi(argv[1]); 
        const double T = std::stod(argv[2]); 
        double J = std::stod(argv[3]);
        double r = std::stod(argv[4]);
        double u = std::stod(argv[5]);
        unsigned long long seed = std::stoull(argv[6]);
        std::string initial_state = argv[7];
        double step_size = std::stod(argv[8]);
        SweepCount adaptation = std::stoull(argv[9]);
        SweepCount settling = std::stoull(argv[10]);
        SweepCount measurement = std::stoull(argv[11]);
        SweepCount interval = std::stoull(argv[12]);
        std::string output_path = argv[13];

        // Initialize model and sampler
        const phi4::Model model{J, r, u}; 
        phi4::Sampler sampler(L, model, T, seed, step_size);
        sampler.initialize(initial_state);
        
        // Store sweep counts
        const SweepCounts sweeps{adaptation, settling, measurement, interval};

        // Validate sweep counts and run
        validate_counts(sweeps);
        if (argc == 15) {
            run_temperature_scan(sampler, sweeps, output_path, argv[14]);
        } else {
            run_chain(sampler, sweeps, output_path);
        }

    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
