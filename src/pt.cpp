// Parallel tempering with local Metropolis updates.
//
// A replica is a field configuration and its random-number stream. It stays on
// the same worker. A temperature slot is a position in the ordered ladder;
// replicas exchange these labels, and output is always sorted by temperature.
//
// The MPI executable owns one replica per process. The serial executable runs
// the same complete ladder in one process, providing a deterministic reference.
#include "phi4/sampler.hpp"
#include "phi4/tempering.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#ifdef PHI4_MPI
#include <mpi.h>
#endif

namespace {
using SweepCount = unsigned long long;

struct RunSettings {
    int lattice_length;
    unsigned long long base_seed;
    std::string initial_state;
    double initial_step_size;
    SweepCount adaptation_sweeps;
    SweepCount settling_sweeps;
    SweepCount measurement_sweeps;
    SweepCount measurement_interval;
    SweepCount exchange_interval;
    phi4::Model model;
    std::vector<double> temperatures;
    std::string output_directory;
};

RunSettings read_run_settings(char **arguments) {
    // scripts/run_pt.py validates the TOML input and passes these resolved values.
    RunSettings settings;
    settings.lattice_length = std::stoi(arguments[1]);
    settings.base_seed = std::stoull(arguments[2]);
    settings.initial_state = arguments[3];
    settings.initial_step_size = std::stod(arguments[4]);
    settings.adaptation_sweeps = std::stoull(arguments[5]);
    settings.settling_sweeps = std::stoull(arguments[6]);
    settings.measurement_sweeps = std::stoull(arguments[7]);
    settings.measurement_interval = std::stoull(arguments[8]);
    settings.exchange_interval = std::stoull(arguments[9]);
    settings.model.J = std::stod(arguments[10]);
    settings.model.r = std::stod(arguments[11]);
    settings.model.u = std::stod(arguments[12]);
    settings.output_directory = arguments[14];

    std::ifstream ladder_file(arguments[13]);
    if (!ladder_file) {
        throw std::invalid_argument("Cannot read temperature ladder");
    }

    double temperature;
    while (ladder_file >> temperature) {
        const bool invalid_temperature = !std::isfinite(temperature) || temperature <= 0;
        const bool not_increasing =
            !settings.temperatures.empty() && temperature <= settings.temperatures.back();
        if (invalid_temperature || not_increasing) {
            throw std::invalid_argument("Require a finite positive increasing ladder");
        }
        settings.temperatures.push_back(temperature);
    }
    if (!ladder_file.eof() || settings.temperatures.empty() || settings.temperatures.size() > 256) {
        throw std::invalid_argument("Require one to 256 temperatures");
    }

    // Each physical replica uses base_seed + replica_id; none may wrap around.
    const auto last_seed_offset = settings.temperatures.size() - 1;
    if (settings.base_seed > std::numeric_limits<unsigned long long>::max() - last_seed_offset) {
        throw std::invalid_argument("Replica seeds overflow");
    }

    constexpr SweepCount maximum_sweeps = 1000000000ULL;
    const bool missing_required_count = settings.settling_sweeps == 0 ||
                                        settings.measurement_sweeps == 0 ||
                                        settings.measurement_interval == 0;
    if (missing_required_count) {
        throw std::invalid_argument("Invalid sampling counts");
    }
    const bool incomplete_measurement_interval =
        settings.measurement_sweeps % settings.measurement_interval != 0;
    const bool too_many_sweeps =
        settings.adaptation_sweeps > maximum_sweeps || settings.settling_sweeps > maximum_sweeps ||
        settings.measurement_sweeps > maximum_sweeps || settings.exchange_interval > maximum_sweeps;
    if (incomplete_measurement_interval || too_many_sweeps) {
        throw std::invalid_argument("Invalid sampling counts");
    }
    return settings;
}

// These named fields keep MPI's packed array layout out of the physics loop.
struct ReplicaMeasurement {
    double total_energy = 0;
    double magnetization = 0;
    double current_step_size = 0;
    double step_size_used_for_sweep = 0;
};

// Own the MPI lifetime and the few collective operations used by this program.
// In serial mode rank zero owns every replica and collectives become local copies.
struct WorkerGroup {
    int rank = 0;
    int size = 1;

    WorkerGroup(int &argument_count, char **&arguments) {
#ifdef PHI4_MPI
        MPI_Init(&argument_count, &arguments);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        (void)argument_count;
        (void)arguments;
#endif
    }

    ~WorkerGroup() {
#ifdef PHI4_MPI
        MPI_Finalize();
#endif
    }

    void gather_replica_measurements(const std::vector<ReplicaMeasurement> &local_measurements,
                                     std::vector<ReplicaMeasurement> &all_measurements) {
#ifdef PHI4_MPI
        // MPI mode has exactly one replica on each rank. Pack four doubles in
        // this documented order, rather than assuming a C++ struct has no padding.
        constexpr int values_per_replica = 4;
        const auto &local = local_measurements.front();
        const double outgoing_values[values_per_replica] = {local.total_energy, local.magnetization,
                                                            local.current_step_size,
                                                            local.step_size_used_for_sweep};
        incoming_values.resize(values_per_replica * all_measurements.size());
        MPI_Gather(outgoing_values, values_per_replica, MPI_DOUBLE, incoming_values.data(),
                   values_per_replica, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // MPI_Gather places rank r's values in replica r's position on rank zero.
        if (rank == 0) {
            for (std::size_t replica_id = 0; replica_id < all_measurements.size(); ++replica_id) {
                const auto offset = values_per_replica * replica_id;
                auto &measurement = all_measurements[replica_id];
                measurement.total_energy = incoming_values[offset];
                measurement.magnetization = incoming_values[offset + 1];
                measurement.current_step_size = incoming_values[offset + 2];
                measurement.step_size_used_for_sweep = incoming_values[offset + 3];
            }
        }
#else
        all_measurements = local_measurements;
#endif
    }

    void broadcast_temperature_labels(std::vector<int> &temperature_of_replica) {
#ifdef PHI4_MPI
        MPI_Bcast(temperature_of_replica.data(), static_cast<int>(temperature_of_replica.size()),
                  MPI_INT, 0, MPI_COMM_WORLD);
#else
        (void)temperature_of_replica;
#endif
    }

    void broadcast_step_sizes(std::vector<double> &step_by_temperature) {
#ifdef PHI4_MPI
        MPI_Bcast(step_by_temperature.data(), static_cast<int>(step_by_temperature.size()),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        (void)step_by_temperature;
#endif
    }

    void abort_all_workers() {
#ifdef PHI4_MPI
        // One rank failing must not leave its peers waiting in a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }

  private:
#ifdef PHI4_MPI
    std::vector<double> incoming_values;
#endif
};

enum class SamplingPhase { Adaptation, Settling, Measurement };

const char *phase_name(SamplingPhase phase) {
    switch (phase) {
    case SamplingPhase::Adaptation:
        return "adapt";
    case SamplingPhase::Settling:
        return "settle";
    case SamplingPhase::Measurement:
        return "measurement";
    }
    throw std::invalid_argument("Unknown sampling phase");
}

std::ofstream open_output_file(const std::string &directory, const std::string &name) {
    std::ofstream file(directory + "/" + name);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << std::setprecision(17);
    return file;
}

class TemperingRun {
  public:
    TemperingRun(const RunSettings &run_settings, WorkerGroup &worker_group)
        : settings(run_settings), workers(worker_group),
          replica_count(static_cast<int>(settings.temperatures.size())), labels(replica_count),
          step_by_temperature(replica_count, settings.initial_step_size),
          all_measurements(replica_count), swap_attempts(replica_count - 1, 0),
          swap_accepts(replica_count - 1, 0), round_trips(replica_count),
          exchange_rng(settings.base_seed ^ 0x9E3779B97F4A7C15ULL) {
#ifdef PHI4_MPI
        if (workers.size != replica_count) {
            throw std::invalid_argument("MPI ranks must equal ladder size");
        }
        first_local_replica = workers.rank;
        past_last_local_replica = workers.rank + 1;
#else
        first_local_replica = 0;
        past_last_local_replica = replica_count;
#endif
        const int local_replica_count = past_last_local_replica - first_local_replica;
        local_measurements.resize(local_replica_count);
        acceptance_sum.resize(local_replica_count, 0);

        // Initially replica r occupies temperature slot r. Its RNG stays with it
        // throughout the run, even after its temperature label changes.
        for (int replica_id = first_local_replica; replica_id < past_last_local_replica;
             ++replica_id) {
            local_samplers.emplace_back(
                settings.lattice_length, settings.model, settings.temperatures[replica_id],
                settings.base_seed + replica_id, settings.initial_step_size);
            local_samplers.back().initialize(settings.initial_state);
        }

        // Use the same arithmetic as the energy-per-site output in the reference.
        site_count = static_cast<double>(settings.lattice_length) * settings.lattice_length *
                     settings.lattice_length;
        if (workers.rank == 0) {
            open_output_files();
        }
    }

    void run() {
        // Adapt without exchanges, freeze one proposal width per temperature,
        // settle with exchanges, then measure with the same fixed proposal widths.
        run_phase(SamplingPhase::Adaptation, settings.adaptation_sweeps);
        run_phase(SamplingPhase::Settling, settings.settling_sweeps);
        run_phase(SamplingPhase::Measurement, settings.measurement_sweeps);

        if (workers.rank == 0) {
            // Explicit close failures must propagate to the launcher.
            production_file.close();
            equilibration_file.close();
            swap_file.close();
            walk_file.close();
            write_summary();
        }
    }

  private:
    const RunSettings &settings;
    WorkerGroup &workers;
    int replica_count;
    int first_local_replica;
    int past_last_local_replica;
    double site_count;

    std::vector<phi4::Sampler> local_samplers;
    phi4::ReplicaLabels labels;
    std::vector<double> step_by_temperature;
    std::vector<double> acceptance_sum;
    std::vector<ReplicaMeasurement> local_measurements;
    std::vector<ReplicaMeasurement> all_measurements;

    std::vector<SweepCount> swap_attempts;
    std::vector<SweepCount> swap_accepts;
    phi4::RoundTrips round_trips;
    SweepCount total_sweeps = 0;
    SweepCount exchange_pass = 0;

    // Exchange decisions use a separate deterministic stream, so they never
    // consume random numbers from the local Metropolis samplers. The fixed XOR
    // mask in the constructor matches the exchange seed recorded by run_pt.py.
    std::mt19937_64 exchange_rng;
    std::uniform_real_distribution<double> uniform_probability{0, 1};

    std::ofstream production_file;
    std::ofstream equilibration_file;
    std::ofstream swap_file;
    std::ofstream walk_file;

    void open_output_files() {
        production_file = open_output_file(settings.output_directory, "measurements.csv");
        equilibration_file = open_output_file(settings.output_directory, "equilibration.csv");
        swap_file = open_output_file(settings.output_directory, "swaps.csv");
        walk_file = open_output_file(settings.output_directory, "walks.csv");

        production_file << "sample_index,sweep,temperature_index,temperature,replica_id,"
                           "energy_per_site,magnetization\n";
        equilibration_file
            << "sweep,phase,temperature_index,temperature,replica_id,energy_per_site,"
               "magnetization,step_size\n";
        swap_file << "sweep,phase,lower_temperature_index,upper_temperature_index,lower_replica,"
                     "upper_replica,lower_total_energy,upper_total_energy,log_ratio,accepted\n";
        walk_file << "sweep,phase,replica_id,temperature_index\n";
    }

    void advance_local_replicas(SweepCount phase_sweep, bool adapt_step_sizes) {
        for (int replica_id = first_local_replica; replica_id < past_last_local_replica;
             ++replica_id) {
            const int local_index = replica_id - first_local_replica;
            auto &sampler = local_samplers[local_index];
            const int temperature_slot = labels.temperature_of_replica[replica_id];
            sampler.temperature = settings.temperatures[temperature_slot];

            // During production, proposal width follows the TEMPERATURE, not
            // the physical replica that happened to visit that temperature.
            if (!adapt_step_sizes) {
                sampler.step_size = step_by_temperature[temperature_slot];
            }
            local_measurements[local_index].step_size_used_for_sweep = sampler.step_size;
            acceptance_sum[local_index] += sampler.sweep();

            // Match the single-chain adaptation: adjust every 50 sweeps toward
            // 40-60% acceptance, with the same bounds and arithmetic order.
            if (adapt_step_sizes && phase_sweep % 50 == 0) {
                const double mean_acceptance = acceptance_sum[local_index] / 50;
                if (mean_acceptance < 0.4) {
                    sampler.step_size *= 0.9;
                } else if (mean_acceptance > 0.6) {
                    sampler.step_size *= 1.1;
                }
                sampler.step_size = std::clamp(sampler.step_size, 1e-4, 100.0);
                acceptance_sum[local_index] = 0;
            }
        }
    }

    void gather_current_measurements() {
        for (int local_index = 0; local_index < static_cast<int>(local_samplers.size());
             ++local_index) {
            const auto &sampler = local_samplers[local_index];
            auto &measurement = local_measurements[local_index];
            measurement.total_energy = phi4::energy(sampler.lattice, sampler.model);
            measurement.magnetization = sampler.magnetization();
            measurement.current_step_size = sampler.step_size;
            if (!std::isfinite(measurement.total_energy) ||
                !std::isfinite(measurement.magnetization)) {
                throw std::runtime_error("Nonfinite replica state");
            }
        }
        workers.gather_replica_measurements(local_measurements, all_measurements);
    }

    void attempt_adjacent_exchanges(SamplingPhase phase) {
        const auto replica_at_temperature = labels.replica_at_temperature();

        // Alternate (0,1),(2,3),... with (1,2),(3,4),... . Each pass contains
        // disjoint pairs, so this inverse mapping stays valid for the whole pass.
        const int first_temperature_slot = static_cast<int>(exchange_pass % 2);
        for (int lower_slot = first_temperature_slot; lower_slot + 1 < replica_count;
             lower_slot += 2) {
            const int upper_slot = lower_slot + 1;
            const int lower_replica = replica_at_temperature[lower_slot];
            const int upper_replica = replica_at_temperature[upper_slot];
            const double lower_total_energy = all_measurements[lower_replica].total_energy;
            const double upper_total_energy = all_measurements[upper_replica].total_energy;

            // Product-canonical Metropolis ratio:
            // log R = (1/T_lower - 1/T_upper) * (H_lower - H_upper).
            // These are TOTAL energies; dividing by N here would change the physics.
            const double log_ratio = phi4::swap_log_ratio(settings.temperatures[lower_slot],
                                                          settings.temperatures[upper_slot],
                                                          lower_total_energy, upper_total_energy);
            const bool accepted = phi4::accept_swap(log_ratio, uniform_probability(exchange_rng));

            // Settling exchanges are logged but excluded from production statistics.
            if (phase == SamplingPhase::Measurement) {
                ++swap_attempts[lower_slot];
                if (accepted) {
                    ++swap_accepts[lower_slot];
                }
            }
            swap_file << total_sweeps << ',' << phase_name(phase) << ',' << lower_slot << ','
                      << upper_slot << ',' << lower_replica << ',' << upper_replica << ','
                      << lower_total_energy << ',' << upper_total_energy << ',' << log_ratio << ','
                      << accepted << '\n';
            if (accepted) {
                labels.exchange(lower_replica, upper_replica);
            }
        }
        ++exchange_pass;

        if (phase == SamplingPhase::Measurement) {
            round_trips.observe(labels.temperature_of_replica);
        }
        for (int replica_id = 0; replica_id < replica_count; ++replica_id) {
            walk_file << total_sweeps << ',' << phase_name(phase) << ',' << replica_id << ','
                      << labels.temperature_of_replica[replica_id] << '\n';
        }
    }

    void write_samples(SamplingPhase phase, SweepCount phase_sweep) {
        // Exchanges happen BEFORE routing this sweep's observations to temperature
        // slots. The gathered H and m stay with their physical configurations.
        const auto replica_at_temperature = labels.replica_at_temperature();
        for (int temperature_slot = 0; temperature_slot < replica_count; ++temperature_slot) {
            const int replica_id = replica_at_temperature[temperature_slot];
            const auto &measurement = all_measurements[replica_id];
            const double temperature = settings.temperatures[temperature_slot];
            const double energy_per_site = measurement.total_energy / site_count;
            if (phase == SamplingPhase::Measurement) {
                const auto sample_index = phase_sweep / settings.measurement_interval - 1;
                production_file << sample_index << ',' << total_sweeps << ',' << temperature_slot
                                << ',' << temperature << ',' << replica_id << ',' << energy_per_site
                                << ',' << measurement.magnetization << '\n';
            } else {
                // Adaptation logs the width actually used for this sweep, even
                // when that width was adjusted immediately after the sweep.
                const double logged_step_size = phase == SamplingPhase::Adaptation
                                                    ? measurement.step_size_used_for_sweep
                                                    : step_by_temperature[temperature_slot];
                equilibration_file << total_sweeps << ',' << phase_name(phase) << ','
                                   << temperature_slot << ',' << temperature << ',' << replica_id
                                   << ',' << energy_per_site << ',' << measurement.magnetization
                                   << ',' << logged_step_size << '\n';
            }
        }
    }

    void freeze_adapted_step_sizes() {
        // No labels moved during adaptation: replica r still has temperature r.
        // At its final sweep, gather_current_measurements saved the UPDATED width,
        // separately from the width logged as used for that last sweep.
        workers.gather_replica_measurements(local_measurements, all_measurements);
        if (workers.rank == 0) {
            for (int temperature_slot = 0; temperature_slot < replica_count; ++temperature_slot) {
                step_by_temperature[temperature_slot] =
                    all_measurements[temperature_slot].current_step_size;
            }
        }
        workers.broadcast_step_sizes(step_by_temperature);
    }

    void run_phase(SamplingPhase phase, SweepCount phase_length) {
        const bool adapt_step_sizes = phase == SamplingPhase::Adaptation;
        if (phase == SamplingPhase::Measurement) {
            // Restart production statistics, but continue the exchange RNG and
            // odd/even pair schedule from settling without resetting them.
            std::fill(swap_attempts.begin(), swap_attempts.end(), 0);
            std::fill(swap_accepts.begin(), swap_accepts.end(), 0);
            if (workers.rank == 0) {
                // The initial production positions establish round-trip endpoints.
                round_trips.observe(labels.temperature_of_replica);
            }
        }

        for (SweepCount phase_sweep = 1; phase_sweep <= phase_length; ++phase_sweep) {
            ++total_sweeps;
            advance_local_replicas(phase_sweep, adapt_step_sizes);

            const bool scheduled_sample = phase_sweep % settings.measurement_interval == 0;
            const bool last_burn_in_sweep =
                phase != SamplingPhase::Measurement && phase_sweep == phase_length;
            const bool record_samples = scheduled_sample || last_burn_in_sweep;
            const bool exchange_due = !adapt_step_sizes && settings.exchange_interval > 0 &&
                                      phase_sweep % settings.exchange_interval == 0;
            if (!record_samples && !exchange_due) {
                continue;
            }

            // Every rank follows the same collective order: gather, exchange on
            // rank zero, broadcast labels, then write temperature-sorted samples.
            gather_current_measurements();
            if (workers.rank == 0 && exchange_due) {
                attempt_adjacent_exchanges(phase);
            }
            if (exchange_due) {
                workers.broadcast_temperature_labels(labels.temperature_of_replica);
            }
            if (workers.rank == 0 && record_samples) {
                write_samples(phase, phase_sweep);
            }
        }

        if (adapt_step_sizes && phase_length > 0) {
            freeze_adapted_step_sizes();
        }
    }

    void write_summary() {
        auto summary = open_output_file(settings.output_directory, "pt_summary.json");
        summary << "{\"replicas\":" << replica_count << ",\"measurement_count_per_temperature\":"
                << settings.measurement_sweeps / settings.measurement_interval
                << ",\"swap_attempts\":[";
        for (std::size_t pair_index = 0; pair_index < swap_attempts.size(); ++pair_index) {
            if (pair_index > 0) {
                summary << ',';
            }
            summary << swap_attempts[pair_index];
        }
        summary << "],\"swap_accepts\":[";
        for (std::size_t pair_index = 0; pair_index < swap_accepts.size(); ++pair_index) {
            if (pair_index > 0) {
                summary << ',';
            }
            summary << swap_accepts[pair_index];
        }
        summary << "],\"round_trips\":[";
        for (int replica_id = 0; replica_id < replica_count; ++replica_id) {
            if (replica_id > 0) {
                summary << ',';
            }
            summary << round_trips.counts[replica_id];
        }
        summary << "],\"step_by_temperature\":[";
        for (int temperature_slot = 0; temperature_slot < replica_count; ++temperature_slot) {
            if (temperature_slot > 0) {
                summary << ',';
            }
            summary << step_by_temperature[temperature_slot];
        }
        summary << "]}\n";
        summary.close();
    }
};
} // namespace

int main(int argc, char **argv) {
    // The launcher checks execution mode before starting MPI processes.
    if (argc == 2 && std::string(argv[1]) == "--build-info") {
#ifdef PHI4_MPI
        std::cout << "{\"execution_mode\":\"mpi\"}\n";
#else
        std::cout << "{\"execution_mode\":\"serial\"}\n";
#endif
        return 0;
    }

    WorkerGroup workers(argc, argv);
    try {
        if (argc != 15) {
            throw std::invalid_argument("Use scripts/run_pt.py CONFIG --binary PATH");
        }
        const auto settings = read_run_settings(argv);
        TemperingRun simulation(settings, workers);
        simulation.run();
    } catch (const std::exception &error) {
        std::cerr << "rank " << workers.rank << ": " << error.what() << '\n';
        workers.abort_all_workers();
        return 1;
    }
}
