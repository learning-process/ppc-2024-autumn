
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovyev_d_convex_hull_binary_image_components_seq {

struct Point {
  int x;
  int y;
  int value;
  double relativeAngle(Point other) const;
};

class Image {
  std::vector<Point> image;
  std::vector<int> components{0};

 public:
  int sizeX;
  int sizeY;

  Image();
  Image(std::vector<int> data, int dimX, int dimY);

  Point getPoint(int x, int y);
  void setPoint(int x, int y, int value);

  std::vector<int> getComponents();
  int newComponent();
  void clearComponents();
  void removeComponent(int n);
  void fixComponents();
};

struct eqUnit {
  int replaceable;
  int replacement;
};

class ConvexHullBinaryImageComponentsSequential : public ppc::core::Task {
 public:
  explicit ConvexHullBinaryImageComponentsSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  Image image;
  std::vector<eqUnit> equivalenceTable;

  static std::vector<int> linearizePoints(std::vector<Point> points) {
    std::vector<int> linear;
    for (size_t i = 0; i < points.size(); i++) {
      linear.push_back(points[i].x);
      linear.push_back(points[i].y);
    }
    return linear;
  }

  std::vector<std::vector<int>> results;
};

}  // namespace solovyev_d_convex_hull_binary_image_components_seq