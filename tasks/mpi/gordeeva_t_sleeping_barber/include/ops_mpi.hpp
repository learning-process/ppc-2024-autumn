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
  int barber_busy;
  std::queue<int> waiting_clients;
  std::vector<int> res;
  std::mutex queue_mutex;
  boost::mpi::communicator world;

  void serve_next_client();
  static void sleep();
  bool add_client_to_queue(int client_id);
};
}  // namespace gordeeva_t_sleeping_barber_mpi
