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

namespace kurakin_m_min_values_by_rows_matrix_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, int count_rows_, int size_rows_)
      : Task(std::move(taskData_)), count_rows(count_rows_), size_rows(size_rows_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_rows;
  int size_rows;
  std::vector<int> input_;
  std::vector<int> res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, int count_rows_, int size_rows_)
      : Task(std::move(taskData_)), count_rows(count_rows_), size_rows(size_rows_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_rows;
  int size_rows;
  std::vector<int> input_, local_input_;
  std::vector<int> res;
  boost::mpi::communicator world;
};

}  // namespace nesterov_a_test_task_mpi