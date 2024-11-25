// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "core/task/include/task.hpp"

namespace koshkin_m_dining_philosophers {

std::vector<int> getRandomVector(int sz);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)){}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  bool request_forks(int philosopher_id);
  bool can_eat(int philosopher_id);
  void release_forks(int philosopher_id);
  void think(int philosopher_id);
  void eat(int philosopher_id);

 private:
  int num_philosophers;
  std::vector<bool> forks;
  std::mutex mutex;
  std::condition_variable condition;
  int eating_philosophers;
  boost::mpi::communicator world;
};

}  // namespace koshkin_m_dining_philosophers