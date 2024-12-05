#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gordeeva_t_sleeping_barber_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int max_waiting_chairs;
  int result;
  boost::mpi::communicator world;

  void barber_logic();
  void dispatcher_logic();
  void client_logic();

  void serve_next_client(int client_id);
};
}  // namespace gordeeva_t_sleeping_barber_mpi
