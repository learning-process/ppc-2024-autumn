#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* input_A;
  int* input_B;
  int column_A{};
  int row_A{};
  int column_B{};
  int row_B{};
  std::vector<int> C;
};

class TestMPITaskParalle : public ppc::core::Task {
 public:
  explicit TestMPITaskParalle(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int column_A{};
  int row_A{};
  int column_B{};
  int row_B{};

  int* C;
  int* input_A;
  int* input_B;
  boost::mpi::communicator world;
};

}  // namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi
