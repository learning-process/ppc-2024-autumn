#include "mpi/tyshkevich_a_monte_carlo/include/ops_mpi.hpp"

namespace tyshkevich_a_monte_carlo_mpi {

bool MonteCarloParallelMPI::validation() {
  internal_order_test();

  return true;
}

bool MonteCarloParallelMPI::pre_processing() {
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

  numPoints = static_cast<int>(precision / world.size());
  if (world.rank() == world.size() - 1) {
    numPoints += static_cast<int>(precision) % world.size();
  }

  globalSum = 0.0;

  return true;
}

bool MonteCarloParallelMPI::run() {
  internal_order_test();

  double localSum = 0.0;
  for (int i = 0; i < numPoints; ++i) {
    std::vector<double> point(dimensions);
    for (int d = 0; d < dimensions; ++d) {
      point[d] = distributions[d](gen);
    }
    localSum += func(point);
  }

  reduce(world, localSum, globalSum, std::plus<>(), 0);

  if (world.rank() == 0) {
    double volume = 1.0;
    for (const auto& bound : bounds) {
      volume *= (bound.second - bound.first);
    }
    result = (volume * globalSum) / precision;
  }

  return true;
}

bool MonteCarloParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }

  return true;
}

}  // namespace tyshkevich_a_monte_carlo_mpi
