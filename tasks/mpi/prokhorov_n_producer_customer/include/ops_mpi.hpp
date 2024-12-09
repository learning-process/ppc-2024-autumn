#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <condition_variable>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_producer_customer_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), ops("default_operation") {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> producer_data;
  std::vector<int> local_input_;
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace prokhorov_n_producer_customer_mpi
