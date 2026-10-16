#include "phi4/model.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <set>

void require(bool x) {
    if (!x)
        throw std::runtime_error("Check failed");
}
void near(double a, double b) {
    require(std::abs(a - b) <= 1e-10 * (1 + std::max(std::abs(a), std::abs(b))));
}
// Deliberately independent: coordinates and explicit positive bonds, no neighbor helper.
double oracle(const std::vector<double> &f, int L, phi4::Model p) {
    auto at = [&](int x, int y, int z) {
        return f.at(static_cast<std::size_t>((z * L + y) * L + x));
    };
    double h = 0;
    for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x) {
                double a = at(x, y, z);
                h += p.r * a * a / 2 + p.u * a * a * a * a / 4;
                h -= p.J * a * at(x == L - 1 ? 0 : x + 1, y, z);
                h -= p.J * a * at(x, y == L - 1 ? 0 : y + 1, z);
                h -= p.J * a * at(x, y, z == L - 1 ? 0 : z + 1);
            }
    return h;
}
int main() {
    phi4::Model p;
    std::mt19937_64 rng(20261015);
    std::uniform_real_distribution<double> draw(-3, 3);
    for (int L : {3, 4, 5}) {
        phi4::Lattice l(L);
        near(phi4::energy(l, p), 0);
        for (std::size_t i = 0; i < l.field.size(); ++i) {
            std::set<std::size_t> seen;
            int x = static_cast<int>(i) % L, y = static_cast<int>(i) / L % L,
                z = static_cast<int>(i) / (L * L);
            for (auto j : l.neighbors[i]) {
                int a = static_cast<int>(j) % L, b = static_cast<int>(j) / L % L,
                    c = static_cast<int>(j) / (L * L);
                auto distance = [L](int v, int w) {
                    int d = std::abs(v - w);
                    return std::min(d, L - d);
                };
                require(distance(x, a) + distance(y, b) + distance(z, c) == 1);
                seen.insert(j);
                require(std::find(l.neighbors[j].begin(), l.neighbors[j].end(), i) !=
                        l.neighbors[j].end());
            }
            require(seen.size() == 6);
        }
        for (double a : {-2.0, 0.0, 1.0, std::sqrt(7.0)}) {
            std::fill(l.field.begin(), l.field.end(), a);
            near(phi4::energy(l, p) / l.field.size(),
                 -3 * p.J * a * a + p.r * a * a / 2 + p.u * a * a * a * a / 4);
        }
        for (auto &a : l.field)
            a = draw(rng);
        near(phi4::energy(l, p), oracle(l.field, L, p));
        const double original = phi4::energy(l, p);
        for (auto &a : l.field)
            a = -a;
        near(phi4::energy(l, p), original);
        auto f = l.field;
        for (int z = 0; z < L; ++z)
            for (int y = 0; y < L; ++y)
                for (int x = 0; x < L; ++x)
                    l.field[l.index(x, y, z)] = f[l.index((x + 1) % L, (y + 1) % L, (z + 1) % L)];
        near(phi4::energy(l, p), original);
        for (std::size_t i = 0; i < l.field.size(); ++i) {
            double b = draw(rng), a = l.field[i], before = oracle(l.field, L, p),
                   dh = phi4::delta_energy(l, p, i, b);
            l.field[i] = b;
            near(dh, oracle(l.field, L, p) - before);
            l.field[i] = a;
        }
    }
    require(phi4::accept(-1, 2, .99));
    require(phi4::accept(0, 2, .99));
    // std::uniform_real_distribution may produce zero; no logarithm is needed.
    require(phi4::accept(2, 2, 0.0));
    require(phi4::accept(2, 2, .36));
    require(!phi4::accept(2, 2, .38));
    require(phi4::accept(2, 4, .60));
    require(!phi4::accept(2, 4, .61));
    bool rejected = false;
    try {
        phi4::Lattice l(2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    rejected = false;
    try {
        phi4::accept(std::numeric_limits<double>::infinity(), 1, .5);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected);
    std::cout << "V1-V4 deterministic checks passed\n";
}
