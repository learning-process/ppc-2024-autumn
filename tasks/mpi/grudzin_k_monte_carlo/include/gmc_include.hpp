#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <numeric>
#include <random>
#include <utility>

#include "core/task/include/task.hpp"
#define functionData double(std::array<double, dimension>&)

namespace grudzin_k_montecarlo_mpi {

template <const int dimension>
class MonteCarloSeq : public ppc::core::Task {
 public:
  explicit MonteCarloSeq(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<functionData> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> dim;
  std::function<functionData> f;
  double result;
  int N;
};

template <const int dimension>
class MonteCarloMpi : public ppc::core::Task {
 public:
  explicit MonteCarloMpi(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<functionData> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> dim;
  std::function<functionData> f;
  double result;
  boost::mpi::communicator world;
};
}  // namespace grudzin_k_montecarlo_mpi