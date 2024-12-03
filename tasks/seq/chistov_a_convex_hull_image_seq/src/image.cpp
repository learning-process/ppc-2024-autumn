#include "seq/chistov_a_convex_hull_image_seq/include/image.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace chistov_a_convex_hull_image_seq_test {

std::vector<int> generateImage(const int width, const int height) {
  if (width <= 0 || height <= 0) {
    return {};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1);

  std::vector<int> image(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image[y * width + x] = dist(gen);
    }
  }

  return image;
}

void printImage(const std::vector<int>& image, const int width, const int height) {
  if (image.empty() || width <= 0 || height <= 0) return;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::cout << image[y * width + x] << " ";
    }
    std::cout << '\n';
  }
}

}  // namespace chistov_a_convex_hull_image_seq_test

namespace chistov_a_convex_hull_image_seq {

std::vector<int> setPoints(const std::vector<Point>& points, const int width, const int height) {
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

void ConvexHull::labelingFirstPass(std::vector<int>& labeled_image) {
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

void ConvexHull::labelingSecondPass(std::vector<int>& labeled_image) {
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

void ConvexHull::processLabeledImage(const std::vector<int>& labeled_image) {
  std::vector<int> component_indices;

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
}

void ConvexHull::labeling() {
  std::vector<int> labeled_image(width * height);

  std::copy(image.begin(), image.end(), labeled_image.begin());
  labelingFirstPass(labeled_image);
  labelingSecondPass(labeled_image);
  processLabeledImage(labeled_image);
}

std::vector<Point> ConvexHull::graham(std::vector<Point> points) {
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

void ConvexHull::find_hull() {
  std::vector<Point> all_points;
  for (const auto& component : components) {
    auto hull = graham(component);
    all_points.insert(all_points.end(), hull.begin(), hull.end());
  }

  std::sort(all_points.begin(), all_points.end(),
            [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  convex_hull = computeConvexHull(all_points);
}

std::vector<Point> ConvexHull::computeConvexHull(const std::vector<Point>& points) {
  if (points.empty()) {
    return points;
  }

  std::vector<Point> lower, upper;
  std::vector<Point> sorted_points = points;

  std::sort(sorted_points.begin(), sorted_points.end(),
            [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  for (const auto& p : sorted_points) {
    while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), *it) <= 0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }

  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());

  return lower;
}

bool ConvexHull::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 2 || taskData->outputs_count.empty() || taskData->inputs[0] == nullptr ||
      taskData->outputs.empty() || taskData->inputs_count[1] <= 0 || taskData->inputs_count[2] <= 0 ||
      taskData->outputs_count[0] <= 0) {
    return false;
  }

  image.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::memcpy(image.data(), tmp_ptr, taskData->inputs_count[0] * sizeof(int));
  if (!std::all_of(image.begin(), image.end(), [](int pixel) { return pixel == 0 || pixel == 1; })) {
    return false;
  }

  return true;
}

bool ConvexHull::pre_processing() {
  internal_order_test();

  size = static_cast<int>(taskData->inputs_count[0]);
  height = static_cast<int>(taskData->inputs_count[1]);
  width = static_cast<int>(taskData->inputs_count[2]);

  return true;
}

bool ConvexHull::run() {
  internal_order_test();

  labeling();
  find_hull();
  image = setPoints(convex_hull, width, height);

  return true;
}

bool ConvexHull::post_processing() {
  internal_order_test();

  std::memcpy(reinterpret_cast<int*>(taskData->outputs[0]), image.data(), image.size() * sizeof(int));

  return true;
}

}  // namespace chistov_a_convex_hull_image_seq
