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

namespace kolokolova_d_radix_integer_merge_sort_mpi {

std::vector<int> radix_sort(std::vector<int>& array);
void counting_sort_radix(std::vector<int>& array, int degree);
std::vector<int> merge_and_sort(const std::vector<int>& vec1, const std::vector<int>& vec2);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
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
  std::vector<int> input_vector, local_vector;
  std::vector<int> res, merge_vec;
  std::vector<int> remaind_vector;
  boost::mpi::communicator world;
  int size_input_vector = 0;
  int local_size = 0;
  int remainder = 0;
};

}  // namespace kolokolova_d_radix_integer_merge_sort_mpi