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

namespace nasedkin_e_matrix_column_max_value_mpi {

std::vector<int> getRandomVector(int sz);

class MatrixColumnMaxTaskSequential : public ppc::core::Task {
 public:
  explicit MatrixColumnMaxTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class MatrixColumnMaxTaskParallel : public ppc::core::Task {
 public:
  explicit MatrixColumnMaxTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)), world() {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace nasedkin_e_matrix_column_max_value_mpi
