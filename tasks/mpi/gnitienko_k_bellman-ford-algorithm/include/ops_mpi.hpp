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

namespace gnitienko_k_bellman_ford_algorithm_mpi {

class BellmanFordAlgSeq : public ppc::core::Task {
 public:
  explicit BellmanFordAlgSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  std::vector<int> shortest_paths;
  const int INF = std::numeric_limits<int>::max();

  bool Iteration(std::vector<int>& paths);
  bool check_negative_cycle();
  void toCRS(const int* input_matrix);
};

class BellmanFordAlgMPI : public ppc::core::Task {
 public:
  explicit BellmanFordAlgMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  size_t values_size{};
  size_t columns_size{};
  size_t row_ptr_size{};
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  std::vector<int> shortest_paths;
  boost::mpi::communicator world;
  const int INF = std::numeric_limits<int>::max();

  bool Iteration(std::vector<int>& paths);
  bool check_negative_cycle();
  void toCRS(const int* input_matrix);
};

}  // namespace gnitienko_k_bellman_ford_algorithm_mpi