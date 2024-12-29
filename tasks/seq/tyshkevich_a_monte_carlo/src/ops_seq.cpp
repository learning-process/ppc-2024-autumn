#include "seq/tyshkevich_a_monte_carlo/include/ops_seq.hpp"

namespace tyshkevich_a_monte_carlo_seq {

bool MonteCarloSequential::validation() {
  internal_order_test();

  return true;
}

bool MonteCarloSequential::pre_processing() {
  internal_order_test();

  dimensions = *reinterpret_cast<int*>(taskData->inputs[0]);
  precision = *reinterpret_cast<double*>(taskData->inputs[1]);
  double left_bound = *reinterpret_cast<double*>(taskData->inputs[2]);
  double right_bound = *reinterpret_cast<double*>(taskData->inputs[3]);

  bounds.resize(dimensions, {left_bound, right_bound});

  std::random_device dev;
  gen.seed(dev());

  distributions.reserve(bounds.size());
  for (const auto& bound : bounds) distributions.emplace_back(bound.first, bound.second);

  globalSum = 0.0;

  return true;
}

bool MonteCarloSequential::run() {
  internal_order_test();

  for (int i = 0; i < static_cast<int>(precision); ++i) {
    std::vector<double> point(dimensions);
    for (int d = 0; d < dimensions; ++d) {
      point[d] = distributions[d](gen);
    }
    globalSum += func(point);
  }

  double volume = 1.0;
  for (const auto& bound : bounds) volume *= (bound.second - bound.first);
  result = (volume * globalSum) / precision;

  return true;
}

bool MonteCarloSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;

  return true;
}

}  // namespace tyshkevich_a_monte_carlo_seq