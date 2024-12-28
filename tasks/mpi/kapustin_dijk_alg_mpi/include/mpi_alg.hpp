#pragma once

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_dijkstras_algorithm_mpi {
class DijkstrasAlgorithmSEQ : public ppc::core::Task {
 public:
  explicit DijkstrasAlgorithmSEQ(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  size_t E{};
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  std::vector<int> res_;
  const int INF = std::numeric_limits<int>::max();

  void CRSconvert(const int* input_matrix);
};
class DijkstrasAlgorithmMPI : public ppc::core::Task {
 public:
  explicit DijkstrasAlgorithmMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  size_t E{};
  size_t values_size{};
  size_t columns_size{};
  size_t row_ptr_size{};
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  std::vector<int> res_;
  boost::mpi::communicator world;
  const int INF = std::numeric_limits<int>::max();
  void CRSconvert(const int* input_matrix);
};

}  // namespace kapustin_dijkstras_algorithm_mpi