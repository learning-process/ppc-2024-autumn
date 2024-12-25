#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_binary_segmentation {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cols;
  int rows;
  std::vector<int> img;
  std::vector<int> mask;
  static void merge_equivalence(std::map<int, std::set<int>>& eq_table, int a, int b);
  static std::vector<int> make_border(const std::vector<int>& img_, int cols_, int rows_);
  static std::vector<int> del_border(const std::vector<int>& img_, int cols_, int rows_);
};
}  // namespace smirnov_i_binary_segmentation