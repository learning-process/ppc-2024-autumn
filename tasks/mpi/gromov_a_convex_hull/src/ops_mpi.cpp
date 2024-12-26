#include "mpi/gromov_a_convex_hull/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace gromov_a_convex_hull_mpi {
std::vector<int> initImage(const std::vector<Point>& points, int width, int height) {
  std::vector<int> grid(width * height, 0);
  if (points.size() < 2) return grid;

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
      grid[minY * width + x] = 1;
    }
    if (maxY >= 0 && maxY < height && x >= 0 && x < width) {
      grid[maxY * width + x] = 1;
    }
  }

  for (int y = minY; y <= maxY; ++y) {
    if (y >= 0 && y < height && minX >= 0 && minX < width) {
      grid[y * width + minX] = 1;
    }
    if (y >= 0 && y < height && maxX >= 0 && maxX < width) {
      grid[y * width + maxX] = 1;
    }
  }
  return grid;
}

int calculateProduct(const Point& a, const Point& b, const Point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

void firstLabel(std::vector<int>& labeled_image, int width, int height) {
  int mark = 2;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int current = labeled_image[i * width + j];
      int left = (j == 0) ? 0 : labeled_image[i * width + j];
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

void secondLabel(std::vector<int>& labeled_image, int width, int height) {
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

std::vector<std::vector<Point>> extractComponents(const std::vector<int>& labeled_image, int width, int height) {
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
  firstLabel(labeled_image, width, height);
  secondLabel(labeled_image, width, height);
  return extractComponents(labeled_image, width, height);
}

std::vector<Point> jarvis(std::vector<Point> points) {
  std::vector<Point> hull;
  if (points.size() < 3) {
    return points;
  }
  int leftmost = 0;
  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i].x < points[leftmost].x || (points[i].x == points[leftmost].x && points[i].y < points[leftmost].y)) {
      leftmost = i;
    }
  }
  int p = leftmost;
  int q;
  do {
    hull.push_back(points[p]);
    q = (p + 1) % points.size();
    for (size_t i = 0; i < points.size(); ++i) {
      if (calculateProduct(points[p], points[q], points[i]) < 0) {
        q = i;
      }
    }
    p = q;
  } while (p != leftmost);
  return hull;
}

bool gromov_a_convex_hull_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size = static_cast<int>(taskData->inputs_count[0]);
    height = static_cast<int>(taskData->inputs_count[1]);
    width = static_cast<int>(taskData->inputs_count[2]);
    components = labeling(grid, width, height);
  }
  return true;
}

bool gromov_a_convex_hull_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count.size() < 2 || taskData->outputs_count.empty() || taskData->inputs[0] == nullptr ||
        taskData->outputs.empty() || taskData->inputs_count[1] <= 0 || taskData->inputs_count[2] <= 0 ||
        taskData->outputs_count[0] <= 0) {
      return false;
    }
    grid.resize(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::memcpy(grid.data(), tmp_ptr, taskData->inputs_count[0] * sizeof(int));
    return std::all_of(grid.begin(), grid.end(), [](int pixel) { return pixel == 0 || pixel == 1; });
  }
  return true;
}

bool gromov_a_convex_hull_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<std::vector<Point>> local_components;
  auto toVector = [](const std::vector<Point>& points) -> std::vector<int> {
    std::vector<int> result(points.size() * 2);
    for (size_t i = 0; i < points.size(); ++i) {
      result[i * 2] = points[i].x;
      result[i * 2 + 1] = points[i].y;
    }
    return result;
  };

  auto toPoints = [](const std::vector<int>& vec) -> std::vector<Point> {
    std::vector<Point> points(vec.size() / 2);
    for (size_t i = 0; i < vec.size(); i += 2) {
      points[i / 2] = {vec[i], vec[i + 1]};
    }
    return points;
  };

  if (world.rank() == 0) {
    int base_count = components.size() / world.size();
    int remainder = components.size() % world.size();
    int offset = 0;
    for (int proc = 1; proc < world.size(); ++proc) {
      int send_count = base_count + (proc < remainder ? 1 : 0);
      world.send(proc, 0, &send_count, 1);
      for (int i = 0; i < send_count; ++i) {
        std::vector<int> int_component = toVector(components[offset + i]);
        world.send(proc, 1, int_component);
      }
      offset += send_count;
    }
    int local_count = base_count + (0 < remainder ? 1 : 0);
    for (int i = 0; i < local_count; ++i) {
      local_components.push_back(components[offset + i]);
    }
  } else {
    int recv_count;
    world.recv(0, 0, &recv_count, 1);
    for (int i = 0; i < recv_count; ++i) {
      std::vector<int> int_component;
      world.recv(0, 1, int_component);
      local_components.push_back(toPoints(int_component));
    }
  }
  std::vector<Point> local_hulls;
  for (const auto& component : local_components) {
    auto hull = jarvis(component);
    local_hulls.insert(local_hulls.end(), hull.begin(), hull.end());
  }
  if (world.rank() == 0) {
    std::vector<Point> merged_hulls;
    merged_hulls.insert(merged_hulls.end(), local_hulls.begin(), local_hulls.end());
    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<int> hull_ints;
      world.recv(proc, 2, hull_ints);
      auto hull = toPoints(hull_ints);
      merged_hulls.insert(merged_hulls.end(), hull.begin(), hull.end());
    }
    grid = initImage(jarvis(merged_hulls), width, height);
  } else {
    std::vector<int> local_hulls_ints = toVector(local_hulls);
    world.send(0, 2, local_hulls_ints);
  }
  return true;
}

bool gromov_a_convex_hull_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::memcpy(reinterpret_cast<int*>(taskData->outputs[0]), grid.data(), grid.size() * sizeof(int));
  }
  return true;
}
}  // namespace gromov_a_convex_hull_mpi