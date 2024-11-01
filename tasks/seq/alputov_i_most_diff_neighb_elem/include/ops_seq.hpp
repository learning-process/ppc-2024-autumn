// Copyright 2024 Alputov Ivan
#pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_most_diff_neighb_elem_seq {

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  int Max_Neighbour_Seq_Pos(const std::vector<int>& data);

 private:
  std::vector<int> inputData;
  int result[2];
};

}  // namespace alputov_i_most_diff_neighb_elem_seq