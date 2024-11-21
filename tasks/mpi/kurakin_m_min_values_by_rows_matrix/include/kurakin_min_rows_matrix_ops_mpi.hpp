#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kurakin_m_min_values_by_rows_matrix_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_rows{};
  int size_rows{};
  std::vector<int> input_;
  std::vector<int> res;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int count_rows{};
  int size_rows{};
  int delta_input{};
  int delta_rows{};
  std::vector<int> input_, local_input_;
  std::vector<int> res;
  boost::mpi::communicator world;
};

}  // namespace kurakin_m_min_values_by_rows_matrix_mpi