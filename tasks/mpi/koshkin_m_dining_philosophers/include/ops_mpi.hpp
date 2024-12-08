#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_m_dining_philosophers {

#define THINKING 1;
#define HUNGRY 2;
#define EATING 3;

class testMpiTaskParallel : public ppc::core::Task {
 public:
  explicit testMpiTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    update_neighbors();
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  bool check_deadlock() noexcept;
  int getStatus() const { return status; }
  void setStatus(int new_status) { status = new_status; }

 private:
  boost::mpi::communicator world;
  int left_philisopher, right_philisopher, status;

  void think();
  void eat();
  void request_forks();
  void release_forks();

  bool check_for_termination();

  void update_neighbors();
};
}  // namespace koshkin_m_dining_philosophers