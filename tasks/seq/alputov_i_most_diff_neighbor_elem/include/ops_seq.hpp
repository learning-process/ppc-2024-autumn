// Copyright 2024 Alputov Ivan
#pragma once

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_most_diff_neighbor_elem_seq {

class MostDiffNeighborElemSeq : public ppc::core::Task {
 public:
  explicit MostDiffNeighborElemSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputVec_;
  std::pair<int, int> maxDifferencePair_;

  static std::pair<int, int> findMaxDifferencePair(const std::vector<int>& vec);
};

}  // namespace alputov_i_most_diff_neighbor_elem_seq