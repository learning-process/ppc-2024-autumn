// Copyright 2024 Nesterov Alexander
#include "seq/poroshin_v_cons_conv_hull_for_bin_image_comp/include/ops_seq.hpp"

bool poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return ((!(taskData->inputs[0] == nullptr) && !(taskData->outputs[0] == nullptr)) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0));
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::run() {
  internal_order_test();
  // Init value for input and output
  int m = taskData->inputs_count[0];
  int n = taskData->inputs_count[1];
  int size = m * n;

  input_.resize(size);

  for (int i = 0; i < size; i++) {
    input_[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }

  std::vector<std::vector<int>> image(m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      image[i].push_back(input_[i * n + j]);
    }
  }

  int count_components =
      poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::label_connected_components(image);
  std::vector<std::vector<std::pair<int, int>>> coords =
      poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::coordinates_connected_components(
          image, count_components);
  for (std::vector<std::pair<int, int>>& t : coords) {
    t = poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::convex_hull(t);
  }

  res.clear();

  for (size_t i = 0; i < coords.size(); i++) {
    for (size_t j = 0; j < coords[i].size(); j++) {
      res.push_back(coords[i][j]);
    }
    res.emplace_back(-1, -1);  // The separating symbol for convex hulls of the connectivity component
  }

  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<std::pair<int, int>*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

std::vector<std::vector<std::pair<int, int>>>
poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::coordinates_connected_components(
    std::vector<std::vector<int>>& labeled_image, int count_components) {
  std::vector<std::vector<std::pair<int, int>>> coords(count_components - 1);
  for (size_t i = 0; i < labeled_image.size(); i++) {
    for (size_t j = 0; j < labeled_image[0].size(); j++) {
      if (labeled_image[i][j] != 0) {
        coords[labeled_image[i][j] - 2].emplace_back(i, j);
      }
    }
  }

  return coords;
}

int poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::label_connected_components(
    std::vector<std::vector<int>>& image) {
  int label = 2;  // Start with 2 to avoid confusion with pixels 0 and 1
  for (size_t i = 0; i < image.size(); i++) {
    for (size_t j = 0; j < image[0].size(); j++) {
      if (image[i][j] == 1) {
        std::stack<std::pair<int, int>> pixelStack;
        pixelStack.emplace(i, j);

        while (!pixelStack.empty()) {
          int x = pixelStack.top().first;
          int y = pixelStack.top().second;
          pixelStack.pop();

          if (x < 0 || x >= static_cast<int>(image.size()) || y < 0 || y >= static_cast<int>(image[0].size()) ||
              image[x][y] != 1) {
            continue;
          }

          image[x][y] = label;

          pixelStack.emplace(x + 1, y);      // Down
          pixelStack.emplace(x - 1, y);      // Up
          pixelStack.emplace(x, y + 1);      // Right
          pixelStack.emplace(x, y - 1);      // Left
          pixelStack.emplace(x - 1, y + 1);  // Top left corner
          pixelStack.emplace(x + 1, y + 1);  // Top right corner
          pixelStack.emplace(x - 1, y - 1);  // Lower left corner
          pixelStack.emplace(x + 1, y - 1);  // Lower right corner
        }
        label++;
      }
    }
  }

  return --label;
}

std::vector<std::pair<int, int>> poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential::convex_hull(
    std::vector<std::pair<int, int>>& inputPoints) {
  auto crossProduct = [](const std::pair<int, int>& origin, const std::pair<int, int>& pointA,
                         const std::pair<int, int>& pointB) {
    return (pointA.first - origin.first) * (pointB.second - origin.second) -
           (pointA.second - origin.second) * (pointB.first - origin.first);
  };

  std::vector<std::pair<int, int>> convexHull;

  if (inputPoints.empty()) {
    return convexHull;
  }

  std::pair<int, int> minPoint = *std::min_element(
      inputPoints.begin(), inputPoints.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
      });

  std::sort(inputPoints.begin(), inputPoints.end(),
            [minPoint, crossProduct](const std::pair<int, int>& pointA, const std::pair<int, int>& pointB) {
              if (crossProduct(minPoint, pointA, pointB) != 0) return crossProduct(minPoint, pointA, pointB) > 0;

              return (pointA.first - minPoint.first) * (pointA.first - minPoint.first) +
                         (pointA.second - minPoint.second) * (pointA.second - minPoint.second) <
                     (pointB.first - minPoint.first) * (pointB.first - minPoint.first) +
                         (pointB.second - minPoint.second) * (pointB.second - minPoint.second);
            });

  convexHull.push_back(minPoint);

  for (size_t i = 1; i < inputPoints.size(); i++) {
    while (convexHull.size() > 1 &&
           crossProduct(convexHull[convexHull.size() - 2], convexHull[convexHull.size() - 1], inputPoints[i]) <= 0) {
      convexHull.pop_back();
    }
    convexHull.push_back(inputPoints[i]);
  }

  convexHull.push_back(convexHull[0]);
  return convexHull;
}