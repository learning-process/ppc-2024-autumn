#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>

#include "core/task/include/task.hpp"

namespace malyshev_v_monte_carlo_integration {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData) : ppc::core::Task(taskData) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void internal_order_test();

 private:
  double a, b, epsilon, res;
  int num_samples;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData)
      : ppc::core::Task(taskData), world(boost::mpi::communicator()) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void internal_order_test();

 private:
  boost::mpi::communicator world;
  double a, b, epsilon, res;
  int num_samples, local_num_samples;
};

double function_square(double x);
double function_constant(double x);
double function_linear(double x);
double function_cubic(double x);

}  // namespace malyshev_v_monte_carlo_integration