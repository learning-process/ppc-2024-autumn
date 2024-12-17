// Copyright 2024 Ivanov Mike
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

#define DELTA 1e-9

namespace ivanov_m_gauss_horizontal_mpi {

int get_linear_index(int row, int col, int number_of_columns);
void swap_rows(std::vector<double>& matrix, int first_row, int second_row, int number_of_columns);
int find_max_row(const std::vector<double>& matrix, int source_row, int source_column, int number_of_rows,
                 int number_of_columns);
int determinant(const std::vector<double>& matrix, int size);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> extended_matrix;  // extended matrix
  int number_of_equations;
  std::vector<double> res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> extended_matrix;  // extended matrix
  int number_of_equations;
  std::vector<double> res;
  boost::mpi::communicator world;
};

}  // namespace ivanov_m_gauss_horizontal_mpi