#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tyshkevich_a_monte_carlo_mpi {

class MonteCarloParallelMPI : public ppc::core::Task {
 public:
  explicit MonteCarloParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_,
                                 std::function<double(const std::vector<double>&)> func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double>&)> func;
  int dimensions, numPoints;
  double precision, globalSum, result = 0.0;

  std::vector<std::pair<double, double>> bounds;
  std::mt19937 gen;
  std::vector<std::uniform_real_distribution<double>> distributions;
  boost::mpi::communicator world;
};

}  // namespace tyshkevich_a_monte_carlo_mpi