#include "seq/sidorina_p_convex_hull_binary_image_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

namespace sidorina_p_convex_hull_binary_image_seq {
double distanceSq(const Point& p1, const Point& p2) { return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2); }

int mix_mult(const Point& p1, const Point& p2, const Point& p3) {
  int dx12 = p2.x - p1.x;
  int dy13 = p3.y - p1.y;
  int dy12 = p2.y - p1.y;
  int dx13 = p3.x - p1.x;

  return dx12 * dy13 - dy12 * dx13;
}

std::vector<int> bin_img(const std::vector<Point>& points, int width, int height) {
  std::vector<int> image(width * height, 0);
  int size = points.size();
  if (size < 2) return image;

  auto [minX, maxX] =
      std::minmax_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.x < b.x; });
  auto [minY, maxY] =
      std::minmax_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.y < b.y; });

  for (int x = minX->x; x <= maxX->x; x++) {
    if (x >= 0 && x < width) {
      image[minY->y * width + x] = 1;
      image[maxY->y * width + x] = 1;
    }
  }
  for (int y = minY->y; y <= maxY->y; y++) {
    if (y >= 0 && y < height) {
      image[y * width + minX->x] = 1;
      image[y * width + maxX->x] = 1;
    }
  }

  return image;
}

void mark_contours(std::vector<int>& image, int width, int height, int num) {
  static int counter = 1;

  const auto& func = [&](int i, int j) -> void {
    int del = image[i * width + j];
    int a = 0;
    int b = 0;  // a - left/right, b - up/low

    if (del == 0) return;

    if (a == 0 && b == 0) {
      image[i * width + j] = counter++;
    } else if (num == 1) {
      image[i * width + j] = std::max(a, b);
    }

    if (j == 0)
      a = 0;
    else
      a = image[i * width + (j - 1)];

    if (i == 0)
      b = 0;
    else
      b = image[(i - 1) * width + j];
  };

  switch (num) {
    case 1:
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) func(i, j);
      }
      break;
    case 2:
      for (int i = height - 1; i >= 0; i--) {
        for (int j = width - 1; j >= 0; j--) func(i, j);
      }
      break;
    default:
      return;
  }
}

std::vector<std::vector<Point>> labeling(const std::vector<int>& image, int width, int height) {
  std::vector<int> label_image(width * height);

  std::copy(image.begin(), image.end(), label_image.begin());
  mark_contours(label_image, width, height, 1);
  mark_contours(label_image, width, height, 2);

  std::unordered_map<int, std::list<Point>> components;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (image[i * width + j] == 0) continue;

      int component_label = image[i * width + j];

      auto& component = components[component_label];
      component.push_back({j, i});
    }
  }
  std::vector<std::vector<Point>> result;
  result.reserve(components.size());
  for (const auto& [label, points] : components) {
    result.emplace_back(points.begin(), points.end());
  }
  return result;
}

std::vector<Point> jarvis(std::vector<Point> points) {
  if (points.empty()) return {};

  struct ComparePoints {
    bool operator()(const Point& a, const Point& b) { return (a.y < b.y) || (a.y == b.y && a.x < b.x); }
  };

  Point min_point = points[0];
  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].x < min_point.x || (points[i].x == min_point.x && points[i].y < min_point.y)) {
      min_point = points[i];
    }
  }

  struct Comparator {
    const Point& min_point;

    Comparator(const Point& min_point) : min_point(min_point) {}

    bool operator()(const Point& p1, const Point& p2) {
      int mult = mix_mult(min_point, p1, p2);
      if (mult != 0) return mult > 0;
      return distanceSq(min_point, p1) < distanceSq(min_point, p2);
    }
  };

  std::sort(points.begin(), points.end(), Comparator(min_point));

  std::vector<Point> hull = {min_point};
  for (size_t i = 1; i < points.size(); i++) {
    while (hull.size() > 1 && mix_mult(hull[hull.size() - 2], hull.back(), points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }

  return hull;
}

bool ConvexHullBinImgSeq::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 2 || taskData->outputs_count.empty() || taskData->inputs[0] == nullptr ||
      taskData->outputs.empty() || taskData->inputs_count[1] <= 0 || taskData->inputs_count[2] <= 0 ||
      taskData->outputs_count[0] <= 0) {
    return false;
  }

  image = std::vector<int>(taskData->inputs_count[0]);

  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], image.data());

  bool is_valid_image = true;

  for (auto i = image.begin(); i != image.end(); ++i) {
    if (*i != 0 && *i != 1) {
      is_valid_image = false;
      break;
    }
  }

  return is_valid_image;
}

bool ConvexHullBinImgSeq::pre_processing() {
  internal_order_test();

  size = static_cast<int>(taskData->inputs_count[0]);
  height = static_cast<int>(taskData->inputs_count[1]);
  width = static_cast<int>(taskData->inputs_count[2]);
  components = labeling(image, width, height);

  return true;
}

bool ConvexHullBinImgSeq::run() {
  internal_order_test();

  std::vector<Point> points;
  for (const auto& component : components) {
    auto hull = jarvis(component);
    for (const auto& point : hull) {
      points.push_back(point);
    }
  }

  image = bin_img(points, width, height);

  return true;
}

bool ConvexHullBinImgSeq::post_processing() {
  internal_order_test();

  std::memcpy(reinterpret_cast<int*>(taskData->outputs[0]), image.data(), image.size() * sizeof(int));

  return true;
}
}  // namespace sidorina_p_convex_hull_binary_image_seq