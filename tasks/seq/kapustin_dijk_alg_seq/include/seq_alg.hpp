#pragma once
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_dijkstra_algorithm {

class DijkstrasAlgorithmSequential : public ppc::core::Task {
 public:
  explicit DijkstrasAlgorithmSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  size_t E{};
  std::vector<int> res_;
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  const int INF = std::numeric_limits<int>::max();

  void CRSconvert(const int* input_matrix);
};

}  // namespace kapustin_i_dijkstra_algorithm
