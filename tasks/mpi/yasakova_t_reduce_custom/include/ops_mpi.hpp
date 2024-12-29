// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_reduce_custom_mpi {
std::vector<std::vector<int>> getRandomMatrix(int rows, int cols);
int CustomReduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  int cols, rows;
  boost::mpi::communicator world;
};

}  // namespace yasakova_t_reduce_custom_mpi