#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_convex_hull_seq {
struct Point {
  int x, y;
};
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> grid;
  std::vector<std::vector<Point>> components;
  int height{};
  int width{};
  int size{};
};

}  // namespace gromov_a_convex_hull_seq