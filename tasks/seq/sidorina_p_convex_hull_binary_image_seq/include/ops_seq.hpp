#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_convex_hull_binary_image_seq {
struct Point {
  int x, y;

  bool operator==(const Point& other) const { return (x == other.x) && (y == other.y); }

  bool operator!=(const Point& other) const { return (x != other.x) || (y != other.y); }

  Point operator-(const Point& other) const { return Point{x - other.x, y - other.y}; }
};

class ConvexHullBinImgSeq : public ppc::core::Task {
 public:
  explicit ConvexHullBinImgSeq(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

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

}  // namespace sidorina_p_convex_hull_binary_image_seq
