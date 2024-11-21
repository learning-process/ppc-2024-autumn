#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_vertikal_shema {

std::vector<int> getRandomMatrix(int rows, int cols);
std::vector<int> getRandomVector(int size);
void calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int num_rows_;
  int num_cols_;
  std::vector<int> distribution;
  std::vector<int> displacement;
  boost::mpi::communicator world;
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), matrix_data_(nullptr), num_rows_(0), num_cols_(0) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* matrix_data_;
  std::vector<std::vector<int>> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int num_rows_;
  int num_cols_;
};

}  // namespace shurigin_s_vertikal_shema
