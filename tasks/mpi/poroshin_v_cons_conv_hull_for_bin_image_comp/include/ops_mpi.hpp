// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>
#include <utility>
#include <stack>
#include <algorithm>

#include "core/task/include/task.hpp"

namespace poroshin_v_cons_conv_hull_for_bin_image_comp_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> gen(int m, int n);  // generate vector (matrix = image)
  static int label_connected_components(std::vector<std::vector<int>>& image);
  static std::vector<std::vector<std::pair<int, int>>> coordinates_�onnected_�omponents(
      std::vector<std::vector<int>>& labeled_image, int count_components);
  static std::vector<std::pair<int, int>> convex_hull(std::vector<std::pair<int, int>>& points);

 private:
  std::vector<int> input_{};
  std::vector<std::pair<int, int>> res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_{};
  std::vector<std::vector<std::pair<int, int>>> local_input_{};
  std::vector<std::pair<int, int>> res{};
  boost::mpi::communicator world;
};

}  // namespace poroshin_v_count_min_val_row_matrix_all_reduce_mpi