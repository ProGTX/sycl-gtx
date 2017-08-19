#pragma once

#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846  // MSDN Math Constants
#endif

namespace ns_random {

inline std::default_random_engine& generator() {
  static std::default_random_engine generator_tmp;
  return generator_tmp;
}

inline double distr(std::default_random_engine& rng) {
  static std::uniform_real_distribution<double> distr_detail(0.0, 1.0);
  return distr_detail(rng);
}

static void reset() {
  generator().seed(0);
}

}  // namespace ns_random

// http://stackoverflow.com/a/27198754
// With modifications to make variables static
static double get_random(uint16_t X[3]) {
  return ns_random::distr(ns_random::generator());
}
