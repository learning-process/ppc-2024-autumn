#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp" 

namespace malyshev_v_mci_mpi {

class MonteCarloMPITask : public ppc::core::Task {
 public:
  explicit MonteCarloMPITask(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(std::function<double(double)> func, double a, double b, int num_samples);

 private:
  double parallel_integration();

  std::function<double(double)> func_;
  double a_;
  double b_;
  int num_samples_;
  double local_result_{};
  double global_result_{};

  boost::mpi::communicator world;
};

}  // namespace malyshev_v_mci_mpi
