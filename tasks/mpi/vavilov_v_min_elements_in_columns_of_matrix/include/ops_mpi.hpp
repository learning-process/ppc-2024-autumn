#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_min_elements_in_columns_of_matrix_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> generate_rand_vec(int size, int lower_bound = 0, int upper_bound = 50);
  static std::vector<std::vector<int>> generate_rand_matr(int rows, int cols);

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<std::vector<int>> local_input_;
  std::vector<int> res_;
  boost::mpi::communicator world;
};

}  // namespace vavilov_v_min_elements_in_columns_of_matrix_mpi
