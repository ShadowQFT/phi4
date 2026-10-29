#include "phi4/sampler.hpp"
#include "phi4/tempering.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#ifdef PHI4_MPI
#include <mpi.h>
#endif

namespace {
struct World {
    int rank = 0, size = 1;
    World(int &argc, char **&argv) {
#ifdef PHI4_MPI
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
        (void)argc;
        (void)argv;
#endif
    }
    ~World() {
#ifdef PHI4_MPI
        MPI_Finalize();
#endif
    }
    void gather(const std::vector<double> &local, std::vector<double> &global) {
#ifdef PHI4_MPI
        MPI_Gather(local.data(), 4, MPI_DOUBLE, global.data(), 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        global = local;
#endif
    }
    void broadcast(std::vector<int> &data) {
#ifdef PHI4_MPI
        MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_INT, 0, MPI_COMM_WORLD);
#else
        (void)data;
#endif
    }
    void broadcast(std::vector<double> &data) {
#ifdef PHI4_MPI
        MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        (void)data;
#endif
    }
    void abort() {
#ifdef PHI4_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
};

std::ofstream output(const std::string &folder, const std::string &name) {
    std::ofstream file(folder + "/" + name);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << std::setprecision(17);
    return file;
}

void run(World &world, char **argv) {
    const int L = std::stoi(argv[1]);
    const auto seed = std::stoull(argv[2]);
    const std::string start = argv[3];
    const double initial_step = std::stod(argv[4]);
    const auto adaptation = std::stoull(argv[5]), settling = std::stoull(argv[6]);
    const auto measurements = std::stoull(argv[7]), interval = std::stoull(argv[8]);
    const auto swap_every = std::stoull(argv[9]);
    const phi4::Model model{std::stod(argv[10]), std::stod(argv[11]), std::stod(argv[12])};
    std::ifstream ladder_file(argv[13]);
    if (!ladder_file)
        throw std::invalid_argument("Cannot read temperature ladder");
    std::vector<double> temperatures;
    double T;
    while (ladder_file >> T) {
        if (!std::isfinite(T) || T <= 0 || (!temperatures.empty() && T <= temperatures.back()))
            throw std::invalid_argument("Require a finite positive increasing ladder");
        temperatures.push_back(T);
    }
    if (!ladder_file.eof() || temperatures.empty() || temperatures.size() > 256)
        throw std::invalid_argument("Require one to 256 temperatures");
    const int count = static_cast<int>(temperatures.size());
    if (seed > std::numeric_limits<unsigned long long>::max() - (count - 1))
        throw std::invalid_argument("Replica seeds overflow");
    if (!settling || !measurements || !interval || measurements % interval ||
        adaptation > 1000000000ULL || settling > 1000000000ULL || measurements > 1000000000ULL ||
        swap_every > 1000000000ULL)
        throw std::invalid_argument("Invalid sampling counts");
#ifdef PHI4_MPI
    if (world.size != count)
        throw std::invalid_argument("MPI ranks must equal ladder size");
    const int first = world.rank, last = world.rank + 1;
#else
    const int first = 0, last = count;
#endif
    std::vector<phi4::Sampler> samplers;
    for (int r = first; r < last; ++r) {
        samplers.emplace_back(L, model, temperatures[r], seed + r, initial_step);
        samplers.back().initialize(start);
    }
    phi4::ReplicaLabels labels(count);
    std::vector<double> step_by_temperature(count, initial_step), acceptance(last - first, 0);
    std::vector<double> local(4 * (last - first)), global(4 * count);
    // Separate, recorded deterministic stream for exchange decisions.
    std::mt19937_64 swap_rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<double> uniform(0, 1);
    const double N = static_cast<double>(L) * L * L;
    std::ofstream production, equilibration, swaps, walks;
    const std::string folder = argv[14];
    if (world.rank == 0) {
        production = output(folder, "measurements.csv");
        equilibration = output(folder, "equilibration.csv");
        swaps = output(folder, "swaps.csv");
        walks = output(folder, "walks.csv");
        production << "sample_index,sweep,temperature_index,temperature,replica_id,energy_per_site,"
                      "magnetization\n";
        equilibration << "sweep,phase,temperature_index,temperature,replica_id,energy_per_site,"
                         "magnetization,step_size\n";
        swaps << "sweep,phase,lower_temperature_index,upper_temperature_index,lower_replica,upper_"
                 "replica,lower_total_energy,upper_total_energy,log_ratio,accepted\n";
        walks << "sweep,phase,replica_id,temperature_index\n";
    }
    std::vector<unsigned long long> attempts(std::max(0, count - 1), 0),
        accepts(attempts.size(), 0);
    phi4::RoundTrips trips(count);
    unsigned long long total_sweeps = 0, exchange_pass = 0;
    auto evolve = [&](const std::string &phase, unsigned long long sweeps, bool adapt) {
        if (phase == "measurement") {
            std::fill(attempts.begin(), attempts.end(), 0);
            std::fill(accepts.begin(), accepts.end(), 0);
            if (world.rank == 0)
                trips.observe(labels.temperature_of_replica);
        }
        for (unsigned long long sweep = 1; sweep <= sweeps; ++sweep) {
            ++total_sweeps;
            for (int r = first; r < last; ++r) {
                auto &sampler = samplers[r - first];
                const int slot = labels.temperature_of_replica[r];
                sampler.temperature = temperatures[slot];
                if (!adapt)
                    sampler.step_size = step_by_temperature[slot];
                const double used_step = sampler.step_size;
                acceptance[r - first] += sampler.sweep();
                local[4 * (r - first) + 3] = used_step;
                if (adapt && sweep % 50 == 0) {
                    const double mean = acceptance[r - first] / 50;
                    if (mean < .4)
                        sampler.step_size *= .9;
                    else if (mean > .6)
                        sampler.step_size *= 1.1;
                    sampler.step_size = std::clamp(sampler.step_size, 1e-4, 100.0);
                    acceptance[r - first] = 0;
                }
            }
            const bool record =
                sweep % interval == 0 || (phase != "measurement" && sweep == sweeps);
            const bool exchange = !adapt && swap_every > 0 && sweep % swap_every == 0;
            if (!record && !exchange)
                continue;
            for (int r = first; r < last; ++r) {
                const auto &sampler = samplers[r - first];
                local[4 * (r - first)] = phi4::energy(sampler.lattice, sampler.model);
                local[4 * (r - first) + 1] = sampler.magnetization();
                local[4 * (r - first) + 2] = sampler.step_size;
                if (!std::isfinite(local[4 * (r - first)]) ||
                    !std::isfinite(local[4 * (r - first) + 1]))
                    throw std::runtime_error("Nonfinite replica state");
            }
            world.gather(local, global);
            if (world.rank == 0 && exchange) {
                const auto inverse = labels.replica_at_temperature();
                for (int slot = static_cast<int>(exchange_pass % 2); slot + 1 < count; slot += 2) {
                    const int a = inverse[slot], b = inverse[slot + 1];
                    const double e_a = global[4 * a], e_b = global[4 * b];
                    const double log_ratio =
                        phi4::swap_log_ratio(temperatures[slot], temperatures[slot + 1], e_a, e_b);
                    const bool accepted = phi4::accept_swap(log_ratio, uniform(swap_rng));
                    if (phase == "measurement") {
                        ++attempts[slot];
                        if (accepted)
                            ++accepts[slot];
                    }
                    swaps << total_sweeps << ',' << phase << ',' << slot << ',' << slot + 1 << ','
                          << a << ',' << b << ',' << e_a << ',' << e_b << ',' << log_ratio << ','
                          << accepted << '\n';
                    if (accepted)
                        labels.exchange(a, b);
                }
                ++exchange_pass;
                if (phase == "measurement")
                    trips.observe(labels.temperature_of_replica);
                for (int r = 0; r < count; ++r)
                    walks << total_sweeps << ',' << phase << ',' << r << ','
                          << labels.temperature_of_replica[r] << '\n';
            }
            if (exchange)
                world.broadcast(labels.temperature_of_replica);
            if (world.rank == 0 && record) {
                const auto inverse = labels.replica_at_temperature();
                for (int slot = 0; slot < count; ++slot) {
                    const int r = inverse[slot];
                    if (phase == "measurement") {
                        production << sweep / interval - 1 << ',' << total_sweeps << ',' << slot
                                   << ',' << temperatures[slot] << ',' << r << ','
                                   << global[4 * r] / N << ',' << global[4 * r + 1] << '\n';
                    } else {
                        equilibration << total_sweeps << ',' << phase << ',' << slot << ','
                                      << temperatures[slot] << ',' << r << ',' << global[4 * r] / N
                                      << ',' << global[4 * r + 1] << ','
                                      << (adapt ? global[4 * r + 3] : step_by_temperature[slot])
                                      << '\n';
                    }
                }
            }
        }
        if (adapt && sweeps > 0) {
            // Labels stayed fixed throughout adaptation. Freeze one step per T.
            world.gather(local, global);
            if (world.rank == 0)
                for (int slot = 0; slot < count; ++slot)
                    step_by_temperature[slot] = global[4 * slot + 2];
            world.broadcast(step_by_temperature);
        }
    };
    evolve("adapt", adaptation, true);
    evolve("settle", settling, false);
    evolve("measurement", measurements, false);
    if (world.rank == 0) {
        production.close();
        equilibration.close();
        swaps.close();
        walks.close();
        auto summary = output(folder, "pt_summary.json");
        summary << "{\"replicas\":" << count
                << ",\"measurement_count_per_temperature\":" << measurements / interval
                << ",\"swap_attempts\":[";
        for (std::size_t i = 0; i < attempts.size(); ++i) {
            if (i)
                summary << ',';
            summary << attempts[i];
        }
        summary << "],\"swap_accepts\":[";
        for (std::size_t i = 0; i < accepts.size(); ++i) {
            if (i)
                summary << ',';
            summary << accepts[i];
        }
        summary << "],\"round_trips\":[";
        for (int r = 0; r < count; ++r) {
            if (r)
                summary << ',';
            summary << trips.counts[r];
        }
        summary << "],\"step_by_temperature\":[";
        for (int slot = 0; slot < count; ++slot) {
            if (slot)
                summary << ',';
            summary << step_by_temperature[slot];
        }
        summary << "]}\n";
        summary.close();
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--build-info") {
#ifdef PHI4_MPI
        std::cout << "{\"execution_mode\":\"mpi\"}\n";
#else
        std::cout << "{\"execution_mode\":\"serial\"}\n";
#endif
        return 0;
    }
    World world(argc, argv);
    try {
        if (argc != 15)
            throw std::invalid_argument("Use scripts/run_pt.py CONFIG --binary PATH");
        run(world, argv);
    } catch (const std::exception &error) {
        std::cerr << "rank " << world.rank << ": " << error.what() << '\n';
        world.abort();
        return 1;
    }
}
