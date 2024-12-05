#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_a_convex_hull_image_mpi_test {
std::vector<int> generateImage(const int width, const int height);
void printImage(const std::vector<int>& image, const int width, const int height);
}  // namespace chistov_a_convex_hull_image_mpi_test

namespace chistov_a_convex_hull_image_mpi {

struct Point {
  int x, y;
};

void labelingFirstPass(std::vector<int>& labeled_image, const int width, const int height);
void labelingSecondPass(std::vector<int>& labeled_image, const int width, const int height);
std::vector<std::vector<Point>> processLabeledImage(const std::vector<int>& labeled_image, const int width,
                                                    const int height);
std::vector<Point> graham(const std::vector<Point> points, const int width, const int height);
std::vector<std::vector<Point>> labeling(const std::vector<int>& image, const int width, const int height);
int cross(const Point& p1, const Point& p2, const Point& p3);
std::vector<int> setPoints(const std::vector<Point>& points, const int width, const int height);

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

class ConvexHullMPI : public ppc::core::Task {
 public:
  explicit ConvexHullMPI(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

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
  boost::mpi::communicator world;
};

}  // namespace chistov_a_convex_hull_image_mpi
