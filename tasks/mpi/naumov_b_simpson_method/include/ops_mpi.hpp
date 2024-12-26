#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_simpson_method_mpi {

using bound_t = std::pair<double, double>;
using func_1d_t = std::function<double(double)>;

double integrir_1d(const func_1d_t& func, const bound_t& bound, int num_steps);

double parallel_integrir_1d(const func_1d_t& func, double lower_bound, double upper_bound, int num_steps,
                            boost::mpi::communicator& world);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data, func_1d_t function, bound_t bounds,
                        int num_steps)
      : Task(std::move(task_data)), bounds_(std::move(bounds)), num_steps_(num_steps), function_(std::move(function)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  bound_t bounds_;
  int num_steps_;
  func_1d_t function_;
  double result_;
  std::vector<bound_t> segments_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> task_data, func_1d_t function, double lower_bound,
                      double upper_bound, int num_steps)
      : Task(std::move(task_data)),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        num_steps_(num_steps),
        function_(std::move(function)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double local_lower_bound_;
  double local_upper_bound_;
  int local_steps_;
  double lower_bound_;
  double upper_bound_;
  int num_steps_;
  func_1d_t function_;
  double local_result_;

  boost::mpi::communicator world;
};

}  // namespace naumov_b_simpson_method_mpi
