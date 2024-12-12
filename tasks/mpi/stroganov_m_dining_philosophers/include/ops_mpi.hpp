// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <condition_variable>
#include <cstring>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace stroganov_m_dining_philosophers {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void eat();
  void think();
  bool distribution_forks();
  void release_forks();
  bool check_deadlock();
  void resolve_deadlock();
  bool check_all_think();

 private:
  boost::mpi::communicator world;
  int status;
  int l_philosopher;
  int r_philosopher;
  int count_philosophers;
};

}  // namespace stroganov_m_dining_philosophers