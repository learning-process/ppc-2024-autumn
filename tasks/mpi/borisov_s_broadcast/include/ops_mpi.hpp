// Copyright 2023 Nesterov Alexander
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

namespace borisov_s_broadcast {

std::vector<double> getRandomMatrix(size_t rows, size_t cols);

class MPITaskMatrixClustering : public ppc::core::Task {
 public:
  explicit MPITaskMatrixClustering(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<double> matrix_;
  std::vector<int> classification_;
  double epsilon_ = 0.1;
  boost::mpi::communicator world;
};

}  // namespace borisov_s_broadcast