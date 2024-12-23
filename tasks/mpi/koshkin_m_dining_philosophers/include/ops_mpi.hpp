// Copyright 2024 Koshkin Matvey

#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_m_dining_philosophers {

#define THINKING 1;
#define HUNGRY 2;
#define EATING 3;

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int wsize = 0;
  int nom = 5;
  int res_ = 0;
  boost::mpi::communicator world;
};

}  // namespace koshkin_m_dining_philosophers