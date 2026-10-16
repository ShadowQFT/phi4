#include "phi4/sampler.hpp"
#include <array>
#include <iostream>
#include <vector>

// Independent Simpson quadrature, not an interacting-model critical-point oracle.
std::array<double, 3> quadrature(int intervals) {
    std::array<double, 4> integral{};
    const double dx = 16.0 / intervals, T = 1.7;
    for (int i = 0; i <= intervals; ++i) {
        const double x = -8 + i * dx, q = x * x;
        const double weight =
            (i == 0 || i == intervals ? 1 : (i % 2 ? 4 : 2)) * std::exp(-(-q / 2 + q * q / 4) / T);
        double power = 1;
        for (auto &v : integral) {
            v += weight * power;
            power *= q;
        }
    }
    return {integral[1] / integral[0], integral[2] / integral[0], integral[3] / integral[0]};
}
int main() {
    auto expected = quadrature(40000), fine = quadrature(80000);
    for (int k = 0; k < 3; ++k)
        if (std::abs(expected[k] - fine[k]) > 1e-10)
            return 1;
    // At |phi|=8 the density is below exp(-580); tail omission is negligible.
    phi4::Sampler sampler(3, {0, -1, 1}, 1.7, 772172, 2);
    sampler.initialize("random");
    for (int i = 0; i < 2000; ++i)
        sampler.sweep();
    std::vector<std::array<double, 3>> blocks;
    for (int block = 0; block < 200; ++block) {
        std::array<double, 3> sums{};
        for (int j = 0; j < 100; ++j) {
            sampler.sweep();
            for (double a : sampler.lattice.field) {
                double q = a * a, power = q;
                for (auto &v : sums) {
                    v += power;
                    power *= q;
                }
            }
        }
        for (auto &v : sums)
            v /= 2700;
        blocks.push_back(sums);
    }
    for (int k = 0; k < 3; ++k) {
        double mean = 0, var = 0;
        for (auto b : blocks)
            mean += b[k] / blocks.size();
        for (auto b : blocks)
            var += (b[k] - mean) * (b[k] - mean);
        double se = std::sqrt(var / (blocks.size() * (blocks.size() - 1)));
        std::cout << "moment " << 2 * (k + 1) << ": " << mean << " expected " << expected[k]
                  << " block SE " << se << '\n';
        if (std::abs(mean - expected[k]) > 6 * se + 1e-9)
            return 1;
    }
}
