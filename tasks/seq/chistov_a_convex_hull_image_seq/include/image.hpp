#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_convex_hull_image_seq_test {
std::vector<int> generateImage(const int width, const int height);
void printImage(const std::vector<int>& image, const int width, const int height);
}  // namespace chistov_a_convex_hull_image_seq_test

namespace chistov_a_convex_hull_image_seq {

struct Point {
  int x, y;
};

int cross(const Point& p1, const Point& p2, const Point& p3);
std::vector<int> setPoints(const std::vector<Point>& points, const int width, const int height);

class ConvexHull : public ppc::core::Task {
 public:
  explicit ConvexHull(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void labeling();
  void find_hull();
 private:
  void labelingFirstPass(std::vector<int>& labeled_image);
  void labelingSecondPass(std::vector<int>& labeled_image);
  void processLabeledImage(const std::vector<int>& labeled_image);
  std::vector<Point> computeConvexHull(const std::vector<Point>& points);
  std::vector<Point> graham(const std::vector<Point> points);

  std::vector<int> image;
  std::vector<std::vector<Point>> components;
  std::vector<Point> convex_hull;
  int width{};
  int height{};
  int size{};
};

}  // namespace chistov_a_convex_hull_image_seq