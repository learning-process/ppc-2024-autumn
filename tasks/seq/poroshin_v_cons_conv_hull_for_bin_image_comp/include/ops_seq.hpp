// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <stack>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_cons_conv_hull_for_bin_image_comp_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> gen(int m, int n);  // generate vector (matrix = image)
  static int label_connected_components(std::vector<std::vector<int>>& image);
  static std::vector<std::vector<std::pair<int, int>>> coordinates_connected_components(
      std::vector<std::vector<int>>& labeled_image, int count_components);
  static std::vector<std::pair<int, int>> convex_hull(std::vector<std::pair<int, int>>& points);

 private:
  std::vector<int> input_;
  std::vector<std::pair<int, int>> res;
};

}  // namespace poroshin_v_cons_conv_hull_for_bin_image_comp_seq