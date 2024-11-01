#pragma once
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

namespace sedova_o_max_of_vector_elements_mpi {
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res_{};
  std::vector<int> input_{};
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  std::vector<int> input_{};
  int res_{};
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string ops;
  boost::mpi::communicator world;
};

std::vector<int> generate_random_vector(size_t size, size_t value);
std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, size_t value);
int find_max_of_matrix(const std::vector<int> matrix);

}  // namespace sedova_o_max_of_vector_elements_mpi