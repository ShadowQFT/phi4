#include "phi4/sampler.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
using SweepCount = unsigned long long;

struct SamplingPlan {
    SweepCount adaptation;
    SweepCount settling;
    SweepCount measurements;
    SweepCount interval;
};

void validate_counts(const SamplingPlan &plan) {
    constexpr SweepCount maximum_sweeps = 1000000000ULL;
    if (plan.interval == 0 || plan.measurements == 0 || plan.settling == 0 ||
        plan.adaptation > maximum_sweeps || plan.settling > maximum_sweeps ||
        plan.measurements > maximum_sweeps) {
        throw std::invalid_argument("Invalid sweep counts");
    }
}

void write_measurement(std::ostream &output, const phi4::Sampler &sampler) {
    // Recompute H rather than accumulating local changes and numerical drift.
    const double energy =
        phi4::energy(sampler.lattice, sampler.model) / sampler.lattice.field.size();
    const double magnetization = sampler.magnetization();
    if (!std::isfinite(energy) || !std::isfinite(magnetization)) {
        throw std::runtime_error("Nonfinite measurement");
    }
    output << energy << ',' << magnetization;
}

// Burn-in samples are written for inspection, but never used as production samples.
void burn_in(phi4::Sampler &sampler, SweepCount count, bool adapt_step, std::ostream &output,
             SweepCount &sweep) {
    double acceptance_sum = 0;
    for (SweepCount j = 1; j <= count; ++j) {
        const double used_step = sampler.step_size;
        const double acceptance = sampler.sweep();
        ++sweep;
        acceptance_sum += acceptance;

        output << sweep << ',';
        write_measurement(output, sampler);
        output << ',' << acceptance << ',' << used_step << ',' << (adapt_step ? "adapt" : "settle")
               << '\n';

        if (adapt_step && j % 50 == 0) {
            const double mean_acceptance = acceptance_sum / 50;
            // Adjust by 10% toward a broad 40-60% acceptance range.
            if (mean_acceptance < 0.4) {
                sampler.step_size *= 0.9;
            } else if (mean_acceptance > 0.6) {
                sampler.step_size *= 1.1;
            }
            sampler.step_size = std::clamp(sampler.step_size, 1e-4, 100.0);
            acceptance_sum = 0;
        }
    }
}

void run_chain(phi4::Sampler &sampler, const SamplingPlan &plan,
               const std::string &output_directory) {
    std::ofstream equilibration(output_directory + "/equilibration.csv");
    std::ofstream measurements(output_directory + "/measurements.csv");
    equilibration.exceptions(std::ios::failbit | std::ios::badbit);
    measurements.exceptions(std::ios::failbit | std::ios::badbit);

    equilibration << std::setprecision(17)
                  << "sweep,energy_per_site,magnetization,acceptance,step_size,phase\n";
    measurements << std::setprecision(17) << "sample_index,sweep,energy_per_site,magnetization\n";

    SweepCount sweep = 0;
    SweepCount sample_count = 0;
    burn_in(sampler, plan.adaptation, true, equilibration, sweep);
    burn_in(sampler, plan.settling, false, equilibration, sweep);
    for (SweepCount j = 1; j <= plan.measurements; ++j) {
        sampler.sweep();
        ++sweep;
        if (j % plan.interval == 0) {
            measurements << sample_count++ << ',' << sweep << ',';
            write_measurement(measurements, sampler);
            measurements << '\n';
        }
    }

    // A close/write failure must reach the launcher as a nonzero exit.
    equilibration.close();
    measurements.close();
    std::cout << std::setprecision(17) << "{\"sample_count\":" << sample_count
              << ",\"final_step_size\":" << sampler.step_size << "}\n";
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
        if (argc != 14) {
            throw std::invalid_argument("Use scripts/run.py CONFIG --binary PATH");
        }

        const int length = std::stoi(argv[1]);
        const double temperature = std::stod(argv[2]);
        const phi4::Model model{std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5])};
        phi4::Sampler sampler(length, model, temperature, std::stoull(argv[6]), std::stod(argv[8]));
        sampler.initialize(argv[7]);

        const SamplingPlan plan{std::stoull(argv[9]), std::stoull(argv[10]), std::stoull(argv[11]),
                                std::stoull(argv[12])};
        validate_counts(plan);
        run_chain(sampler, plan, argv[13]);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
