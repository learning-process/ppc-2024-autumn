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

namespace solovyev_d_vector_max_mpi {

int vectorMax(std::vector<int, std::allocator<int>> v);

class VectorMaxSequential : public ppc::core::Task {
 public:
  explicit VectorMaxSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data;
  int result{};
  std::string ops;
};

class VectorMaxMPIParallel : public ppc::core::Task {
 public:
  explicit VectorMaxMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data, localData;
  int result{};
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace solovyev_d_vector_max_mpi