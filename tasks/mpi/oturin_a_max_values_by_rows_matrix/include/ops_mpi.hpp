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

namespace oturin_a_max_values_by_rows_matrix_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t n = 0;
  size_t m = 0;
  std::vector<int> input_;
  std::vector<int> res;
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  /*
        m           maxes:
        ^
        | -9 99    :  99
        | 12 06    :  12
        +------> n
  */
  size_t n = 0;
  size_t m = 0;
  std::vector<int> input_, local_input_;
  std::vector<int> res;
  std::string ops;

  int world_size = -1;
  int world_rank = -1;
};

}  // namespace oturin_a_max_values_by_rows_matrix_mpi