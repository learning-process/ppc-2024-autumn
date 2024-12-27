#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_convex_hull_image_seq {
struct Point {
  int x, y;
};

class ConvexHullSEQ : public ppc::core::Task {
 public:
  explicit ConvexHullSEQ(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image;
  std::vector<std::vector<Point>> components;
  int width{};
  int height{};
  int size{};
};

}  // namespace chistov_a_convex_hull_image_seq