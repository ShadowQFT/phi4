#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace phi4 {
struct Model {
    double J = 1;
    double r = -1;
    double u = 1;

    void validate() const {
        // J=0 supports the independent on-site distribution validation.
        if (!std::isfinite(J) || !std::isfinite(r) || !std::isfinite(u) || J < 0 || r >= 0 ||
            u <= 0) {
            throw std::invalid_argument("Require finite J>=0, r<0, u>0");
        }
    }
};
class Lattice {
  public:
    explicit Lattice(int length) : L(length) {
        // L=1/2 need different self-bond or repeated-neighbor conventions.
        if (L < 3 || L > 512) {
            throw std::invalid_argument("L must be in [3,512]");
        }
        const auto n = static_cast<std::size_t>(L) * L * L;
        field.resize(n);
        neighbors.resize(n);
        for (int z = 0; z < L; ++z) {
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    // Pair order is +x,-x,+y,-y,+z,-z throughout the model.
                    neighbors[index(x, y, z)] = {
                        index((x + 1) % L, y, z), index((x + L - 1) % L, y, z),
                        index(x, (y + 1) % L, z), index(x, (y + L - 1) % L, z),
                        index(x, y, (z + 1) % L), index(x, y, (z + L - 1) % L)};
                }
            }
        }
    }

    std::size_t index(int x, int y, int z) const {
        return x + static_cast<std::size_t>(L) * (y + L * z);
    }

    int L;
    std::vector<double> field;
    std::vector<std::array<std::size_t, 6>> neighbors;
};

inline double onsite(double a, const Model &p) {
    const double squared = a * a;
    return 0.5 * p.r * squared + 0.25 * p.u * squared * squared;
}

inline double energy(const Lattice &lattice, const Model &model) {
    double total = 0;
    for (std::size_t i = 0; i < lattice.field.size(); ++i) {
        const double field = lattice.field[i];
        // Positive directions count each undirected bond exactly once.
        const double positive_neighbors = lattice.field[lattice.neighbors[i][0]] +
                                          lattice.field[lattice.neighbors[i][2]] +
                                          lattice.field[lattice.neighbors[i][4]];
        total += onsite(field, model) - model.J * field * positive_neighbors;
    }
    return total;
}

inline double delta_energy(const Lattice &lattice, const Model &model, std::size_t site,
                           double proposed) {
    double neighbor_sum = 0;
    for (auto neighbor : lattice.neighbors.at(site)) {
        neighbor_sum += lattice.field[neighbor];
    }
    const double current = lattice.field.at(site);
    // All six incident bonds change once; there is no extra factor of 1/2.
    return -model.J * (proposed - current) * neighbor_sum + onsite(proposed, model) -
           onsite(current, model);
}

inline bool accept(double delta_h, double temperature, double uniform) {
    if (!std::isfinite(delta_h) || !std::isfinite(temperature) || temperature <= 0 ||
        !(uniform >= 0 && uniform < 1)) {
        throw std::runtime_error("Invalid acceptance inputs");
    }
    // Downhill moves are always accepted. Uphill moves use the Boltzmann probability.
    if (delta_h <= 0) {
        return true;
    }
    const double probability = std::exp(-delta_h / temperature);
    return uniform < probability;
}
} // namespace phi4
