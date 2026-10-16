#pragma once
#include "phi4/model.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <string>

namespace phi4 {
class Sampler {
  public:
    Sampler(int length, Model parameters, double temperature_value, std::uint64_t seed, double step)
        : lattice(length), model(parameters), temperature(temperature_value), step_size(step),
          rng(seed) {
        model.validate();
        if (!std::isfinite(temperature) || temperature <= 0 || !std::isfinite(step_size) ||
            step_size <= 0) {
            throw std::invalid_argument("Require temperature>0 and step_size>0");
        }
    }

    void initialize(const std::string &start) {
        if (start == "random") {
            std::uniform_real_distribution<double> initial_field(-1.0, 1.0);
            for (auto &field : lattice.field) {
                field = initial_field(rng);
            }
        } else if (start == "ordered_positive" || start == "ordered_negative") {
            // Interacting uniform minima differ from the isolated on-site minima.
            const double amplitude = std::sqrt((6 * model.J - model.r) / model.u) *
                                     (start == "ordered_positive" ? 1 : -1);
            std::fill(lattice.field.begin(), lattice.field.end(), amplitude);
        } else {
            throw std::invalid_argument("Invalid start");
        }
    }

    double sweep() {
        std::size_t accepted = 0;
        std::uniform_real_distribution<double> displacement(-step_size, step_size);
        std::uniform_real_distribution<double> probability(0.0, 1.0);

        // Monte Carlo Loop
        for (std::size_t i = 0; i < lattice.field.size(); ++i) {
            // propose new field value
            const double proposed = lattice.field[i] + displacement(rng); 

            // compute energy change for the proposed update
            const double delta_h = delta_energy(lattice, model, i, proposed);

            // accept with boltzmann probability
            if (accept(delta_h, temperature, probability(rng))) {
                lattice.field[i] = proposed;
                ++accepted;
            }
        }
        // return acceptance rate for this sweep
        return static_cast<double>(accepted) / lattice.field.size();
    }

    double magnetization() const {
        // m = (1/N) * sum_i phi_i
        double total = 0;
        for (double field : lattice.field) {
            total += field;
        }
        return total / lattice.field.size();
    }

    Lattice lattice;
    Model model;
    double temperature;
    double step_size;

  private:
    std::mt19937_64 rng;
};
} // namespace phi4
