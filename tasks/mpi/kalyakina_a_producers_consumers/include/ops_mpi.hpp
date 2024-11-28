// Copyright 2023 Nesterov Alexander
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

namespace kalyakina_a_producers_consumers_mpi {

class ProducersConsumersTaskParallel : public ppc::core::Task {
  static int ProducersFunction();
  static void ConsumersFunction(int data);

 public:
  explicit ProducersConsumersTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  unsigned int buffer_size{};
  int producers_count{};
  int sources{};
  int result{};
  boost::mpi::communicator world;
};

}  // namespace kalyakina_a_producers_consumers_mpi