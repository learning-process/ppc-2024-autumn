// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
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
  bool distribution_forks(int philosopher_id);
  void release_forks(int philosopher_id);
  bool can_eat(int philosopher_id);
  void eat(int philosopher_id);
  void think(int philosopher_id);
  const std::vector<bool>& get_forks() const;

 private:
  int count_philosophers;
  std::vector<bool> forks;
  std::mutex mutex;
  std::condition_variable status;
  int dining_philosophers;
  boost::mpi::communicator world;
};

}  // namespace stroganov_m_dining_philosophers