#include "seq/chistov_a_convex_hull_image_seq/include/image.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace chistov_a_convex_hull_image_seq {
std::vector<int> setPoints(const std::vector<Point>& points, int width, int height) {
  std::vector<int> image(width * height, 0);
  if (points.size() < 2) return image;

  int minX = std::min(points[0].x, points[1].x);
  int maxX = std::max(points[0].x, points[1].x);
  int minY = std::min(points[0].y, points[1].y);
  int maxY = std::max(points[0].y, points[1].y);

  for (size_t i = 2; i < points.size(); ++i) {
    minX = std::min(minX, points[i].x);
    maxX = std::max(maxX, points[i].x);
    minY = std::min(minY, points[i].y);
    maxY = std::max(maxY, points[i].y);
  }

  for (int x = minX; x <= maxX; ++x) {
    if (minY >= 0 && minY < height && x >= 0 && x < width) {
      image[minY * width + x] = 1;
    }
    if (maxY >= 0 && maxY < height && x >= 0 && x < width) {
      image[maxY * width + x] = 1;
    }
  }

  for (int y = minY; y <= maxY; ++y) {
    if (y >= 0 && y < height && minX >= 0 && minX < width) {
      image[y * width + minX] = 1;
    }
    if (y >= 0 && y < height && maxX >= 0 && maxX < width) {
      image[y * width + maxX] = 1;
    }
  }

  return image;
}

int cross(const Point& p1, const Point& p2, const Point& p3) {
  return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
}

void labelingFirstPass(std::vector<int>& labeled_image, int width, int height) {
  int mark = 2;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int current = labeled_image[i * width + j];
      int left = (j == 0) ? 0 : labeled_image[i * width + (j - 1)];
      int upper = (i == 0) ? 0 : labeled_image[(i - 1) * width + j];

      if (current == 0) continue;

      if (left == 0 && upper == 0) {
        labeled_image[i * width + j] = mark++;
      } else {
        labeled_image[i * width + j] = std::max(left, upper);
      }
    }
  }
}

void labelingSecondPass(std::vector<int>& labeled_image, int width, int height) {
  for (int i = height - 1; i >= 0; --i) {
    for (int j = width - 1; j >= 0; --j) {
      int current = labeled_image[i * width + j];
      int right = (j == width - 1) ? 0 : labeled_image[i * width + (j + 1)];
      int lower = (i == height - 1) ? 0 : labeled_image[(i + 1) * width + j];

      if (current == 0 || (right == 0 && lower == 0)) continue;

      labeled_image[i * width + j] = std::max(right, lower);
    }
  }
}

std::vector<std::vector<Point>> processLabeledImage(const std::vector<int>& labeled_image, int width, int height) {
  std::vector<int> component_indices;
  std::vector<std::vector<Point>> components;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (labeled_image[i * width + j] == 0) continue;

      int component_label = labeled_image[i * width + j];

      if (static_cast<size_t>(component_label) < component_indices.size() && component_indices[component_label] != -1) {
        components[component_indices[component_label]].push_back(Point{j, i});
      } else {
        component_indices.resize(std::max(component_indices.size(), static_cast<size_t>(component_label + 1)), -1);
        component_indices[component_label] = components.size();
        components.push_back({Point{j, i}});
      }
    }
  }
  return components;
}

std::vector<std::vector<Point>> labeling(const std::vector<int>& image, int width, int height) {
  std::vector<int> labeled_image(width * height);

  std::copy(image.begin(), image.end(), labeled_image.begin());
  labelingFirstPass(labeled_image, width, height);
  labelingSecondPass(labeled_image, width, height);
  return processLabeledImage(labeled_image, width, height);
}

std::vector<Point> graham(std::vector<Point> points) {
  std::vector<Point> hull;

  if (points.empty()) {
    return hull;
  }

  Point min = *std::min_element(points.begin(), points.end(),
                                [](const Point& a, const Point& b) { return a.x == b.x ? a.y < b.y : a.x < b.x; });

  std::sort(points.begin(), points.end(), [min](const Point& p1, const Point& p2) {
    if (cross(min, p1, p2) != 0) return cross(min, p1, p2) > 0;

    return (p1.x - min.x) * (p1.x - min.x) + (p1.y - min.y) * (p1.y - min.y) <
           (p2.x - min.x) * (p2.x - min.x) + (p2.y - min.y) * (p2.y - min.y);
  });

  hull.push_back(min);

  for (size_t i = 1; i < points.size(); ++i) {
    while (hull.size() > 1 && cross(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  return hull;
}

bool ConvexHullSEQ::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 2 || taskData->outputs_count.empty() || taskData->inputs[0] == nullptr ||
      taskData->outputs.empty() || taskData->inputs_count[1] <= 0 || taskData->inputs_count[2] <= 0 ||
      taskData->outputs_count[0] <= 0) {
    return false;
  }

  image.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::memcpy(image.data(), tmp_ptr, taskData->inputs_count[0] * sizeof(int));

  return std::all_of(image.begin(), image.end(), [](int pixel) { return pixel == 0 || pixel == 1; });
}

bool ConvexHullSEQ::pre_processing() {
  internal_order_test();

  size = static_cast<int>(taskData->inputs_count[0]);
  height = static_cast<int>(taskData->inputs_count[1]);
  width = static_cast<int>(taskData->inputs_count[2]);
  components = labeling(image, width, height);

  return true;
}

bool ConvexHullSEQ::run() {
  internal_order_test();

  std::vector<Point> points;
  for (const auto& component : components) {
    auto hull = graham(component);
    points.insert(points.end(), hull.begin(), hull.end());
  }

  image = setPoints(points, width, height);

  return true;
}

bool ConvexHullSEQ::post_processing() {
  internal_order_test();

  std::memcpy(reinterpret_cast<int*>(taskData->outputs[0]), image.data(), image.size() * sizeof(int));

  return true;
}
}  // namespace chistov_a_convex_hull_image_seq
