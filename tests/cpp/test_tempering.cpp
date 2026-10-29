#include "phi4/tempering.hpp"
#include <array>
#include <iostream>

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    // Independent Boltzmann-product oracle on a finite two-replica state space.
    const double cold = 1.25, hot = 2.5;
    const std::array<double, 3> energies{-3., 0., 4.};
    for (double a : energies)
        for (double b : energies) {
            const double forward = std::min(1., std::exp(phi4::swap_log_ratio(cold, hot, a, b)));
            const double reverse = std::min(1., std::exp(phi4::swap_log_ratio(cold, hot, b, a)));
            const double original = std::exp(-a / cold - b / hot);
            const double exchanged = std::exp(-b / cold - a / hot);
            require(std::abs(original * forward - exchanged * reverse) < 1e-12,
                    "Replica exchange violates product-ensemble detailed balance");
        }
    require(std::abs(phi4::swap_log_ratio(2, 4, 10, 14) + 1) < 1e-14,
            "Total-energy swap exponent has wrong sign or temperature factors");
    require(phi4::accept_swap(-1, .1) && !phi4::accept_swap(-1, .9), "Swap acceptance oracle");
    require(phi4::accept_swap(-1000, 0) && phi4::accept_swap(0, .999), "Swap limits");
    bool rejected = false;
    try {
        phi4::swap_log_ratio(2, 1, 0, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "Invalid ladder not rejected");
    phi4::ReplicaLabels labels(4);
    labels.exchange(0, 1);
    labels.exchange(2, 3);
    require(labels.replica_at_temperature() == std::vector<int>({1, 0, 3, 2}),
            "Odd/even permutation");
    labels.temperature_of_replica[0] = labels.temperature_of_replica[1];
    rejected = false;
    try {
        labels.replica_at_temperature();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "Duplicate temperature label not rejected");
    phi4::RoundTrips trips(2);
    trips.observe({0, 1});
    trips.observe({1, 0});
    require(trips.counts[0] == 0, "One-way passage counted as round trip");
    trips.observe({0, 1});
    trips.observe({0, 1});
    require(trips.counts == std::vector<unsigned long long>({1, 1}), "Round-trip counting");
    trips.observe({1, 0});
    trips.observe({0, 1});
    require(trips.counts[0] == 2, "Repeated round-trip counting");
    std::cout
        << "Exchange detailed balance, total-energy convention, labels and round trips passed\n";
}
