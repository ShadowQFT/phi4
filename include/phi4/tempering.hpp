#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace phi4 {
// Energies are TOTAL H, never H/N. The ensemble is exp(-H/T).
inline double swap_log_ratio(double lower_T, double upper_T, double lower_H, double upper_H) {
    if (!(lower_T > 0 && upper_T > lower_T) || !std::isfinite(upper_T) || !std::isfinite(lower_H) ||
        !std::isfinite(upper_H)) {
        throw std::invalid_argument("Invalid replica-exchange temperatures or total energies");
    }
    const double value = (1 / lower_T - 1 / upper_T) * (lower_H - upper_H);
    if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite swap log ratio");
    return value;
}

inline bool accept_swap(double log_ratio, double uniform) {
    if (!std::isfinite(log_ratio) || !(uniform >= 0 && uniform < 1))
        throw std::invalid_argument("Invalid swap acceptance input");
    return log_ratio >= 0 || uniform == 0 || std::log(uniform) < log_ratio;
}

// A rank keeps its field and RNG. Only its temperature label moves, as in J1J2.
struct ReplicaLabels {
    std::vector<int> temperature_of_replica;
    explicit ReplicaLabels(int count) : temperature_of_replica(count) {
        if (count < 1)
            throw std::invalid_argument("Empty ladder");
        std::iota(temperature_of_replica.begin(), temperature_of_replica.end(), 0);
    }
    std::vector<int> replica_at_temperature() const {
        std::vector<int> inverse(temperature_of_replica.size(), -1);
        for (std::size_t r = 0; r < temperature_of_replica.size(); ++r) {
            const int slot = temperature_of_replica[r];
            if (slot < 0 || slot >= static_cast<int>(inverse.size()) || inverse[slot] != -1)
                throw std::runtime_error("Replica labels are not a permutation");
            inverse[slot] = static_cast<int>(r);
        }
        return inverse;
    }
    void exchange(int a, int b) {
        std::swap(temperature_of_replica.at(a), temperature_of_replica.at(b));
    }
};

struct RoundTrips {
    std::vector<int> home, stage;
    std::vector<unsigned long long> counts;
    explicit RoundTrips(int count) : home(count, -1), stage(count, 0), counts(count, 0) {}
    void observe(const std::vector<int> &labels) {
        const int last = static_cast<int>(labels.size()) - 1;
        if (last < 1)
            return;
        for (std::size_t r = 0; r < labels.size(); ++r) {
            const int slot = labels[r];
            if (slot != 0 && slot != last)
                continue;
            if (home[r] == -1) {
                home[r] = slot;
                stage[r] = 1;
            } else if (stage[r] == 1 && slot != home[r])
                stage[r] = 2;
            else if (stage[r] == 2 && slot == home[r]) {
                ++counts[r];
                stage[r] = 1;
            }
        }
    }
};
} // namespace phi4
