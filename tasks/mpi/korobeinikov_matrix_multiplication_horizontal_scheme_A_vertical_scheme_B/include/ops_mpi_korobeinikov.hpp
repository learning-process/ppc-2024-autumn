// Copyright 2024 Korobeinikov Arseny
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

namespace korobeinikov_a_test_task_mpi_lab_02 {

struct Matrix {
  std::vector<int> data;
  int count_rows;
  int count_cols;

  Matrix() {
    count_rows = 0;
    count_cols = 0;
    data = std::vector<int>();
  }

  Matrix(int count_rows_, int count_cols_) {
    count_rows = count_rows_;
    count_cols = count_cols_;
    data = std::vector<int>(count_cols * count_rows);
  }

  int& get_el(int row, int col) {
    size_t index = row * count_cols + col;
    assert(index < data.size());
    return data[index];
  }

  const int& get_el(int row, int col) const {
    size_t index = row * count_cols + col;
    assert(index < data.size());
    return data[index];
  }
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix A;
  Matrix B;

  Matrix res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  Matrix A;
  Matrix B;
  std::vector<int> local_A_rows;
  std::vector<int> local_B_cols;

  Matrix res;
  boost::mpi::communicator world;
};

}  // namespace korobeinikov_a_test_task_mpi_lab_02