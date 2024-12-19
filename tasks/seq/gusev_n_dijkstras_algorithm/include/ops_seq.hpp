// Copyright 2024 Nesterov Alexander
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_dijkstras_algorithm_seq {

class DijkstrasAlgorithmSequential : public ppc::core::Task {
 public:
  explicit DijkstrasAlgorithmSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  struct SparseGraphCRS {
    int num_vertices;
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<double> values;
  };

 private:
};

}  // namespace gusev_n_dijkstras_algorithm_seq