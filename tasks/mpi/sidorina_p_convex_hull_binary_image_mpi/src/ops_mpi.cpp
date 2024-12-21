#include "mpi/sidorina_p_convex_hull_binary_image_mpi/include/ops_mpi.hpp"

namespace sidorina_p_convex_hull_binary_image_mpi {
double distanceSq(const Point& p1, const Point& p2) { return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2); }

double mix_mult(const Point& p1, const Point& p2, const Point& p3) {
  int dx12 = p2.x - p1.x, dy13 = p3.y - p1.y, dy12 = p2.y - p1.y, dx13 = p3.x - p1.x;

  return dx12 * dy13 - dy12 * dx13;
}

int count_rem(int rem, int i) {
  if (i < rem)
      return 1;
  else
      return 0;
}

std::vector<int> conv_vec(const std::vector<Point>& points) {
  std::vector<int> result(points.size() * 2);
  for (size_t i = 0; i < points.size(); i++) {
    result[i * 2] = points[i].x;
    result[i * 2 + 1] = points[i].y;
  }
  return result;
}

std::vector<Point> conv_point(const std::vector<int>& vec) {
  std::vector<Point> points = {};
  points.reserve(vec.size() / 2);
  for (auto i = vec.begin(); i != vec.end(); ++i, ++i) {
    if (i + 1 != vec.end()) {
      points.push_back({*i, *(i + 1)});
    }
  }
  return points;
}

std::vector<int> bin_img(const std::vector<Point>& points, int width, int height) {
  std::vector<int> image(width * height, 0);
  int size = points.size();
  if (size < 2) return image;

  int mX = points[0].x, MX = points[0].x, mY = points[0].y, MY = points[0].y;
  for (int i = 1; i < size; i++) {
    mX = std::min(mX, points[i].x);
    MX = std::max(MX, points[i].x);
    mY = std::min(mY, points[i].y);
    MY = std::max(MY, points[i].y);
  }

  for (int x = mX; x <= MX; x++) {
    if (x >= 0 && x < width) {
      image[mY * width + x] = 1;
      image[MY * width + x] = 1;
    }
  }
  for (int y = mY; y <= MY; y++) {
    if (y >= 0 && y < height) {
      image[y * width + mX] = 1;
      image[y * width + MX] = 1;
    }
  }

  return image;
}

void mark_contours(std::vector<int>& image, int width, int height, int num) {
  static int counter = 1;

  const auto& func = [&](int i, int j) -> void {
    int del = image[i * width + j];
    int a = 0, b = 0;  // a - left/right, b - up/low

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
  for (const auto& [label, points] : components) {
    result.push_back({points.begin(), points.end()});
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

bool ConvexHullBinImgMpi::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() < 2 || taskData->outputs_count.empty() || taskData->inputs[0] == nullptr ||
        taskData->outputs.empty() || taskData->inputs_count[1] <= 0 || taskData->inputs_count[2] <= 0 ||
        taskData->outputs_count[0] <= 0) {
      return false;
    }

    image = std::vector<int>(taskData->inputs_count[0]);

    std::memcpy(image.data(), reinterpret_cast<int*>(taskData->inputs[0]), taskData->inputs_count[0] * sizeof(int));

    bool is_valid_image = true;

    for (auto i = image.begin(); i != image.end(); i++) {
      if (*i != 0 && *i != 1) {
        is_valid_image = false;
        break;
      }
    }

    return is_valid_image;
  }
  return true;
}

bool ConvexHullBinImgMpi::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size = static_cast<int>(taskData->inputs_count[0]);
    height = static_cast<int>(taskData->inputs_count[1]);
    width = static_cast<int>(taskData->inputs_count[2]);
    components = labeling(image, width, height);
  }

  return true;
}

bool ConvexHullBinImgMpi::run() {
  internal_order_test();

  std::vector<std::vector<Point>> local_components;
  int c_size = components.size();
  int w_size = world.size();
  int count = c_size / w_size;
  int rem = c_size % w_size;
  local_components.resize(w_size);
  for (int i = 0; i < w_size; ++i) {
    if (i < rem) {
      local_components[i].reserve(count + 1);
    } else {
      local_components[i].reserve(count);
    }
  }
  if (world.rank() == 0) {
    MPI_Scatterv(components.data(), &count, &rem, MPI_INT, local_components[0].data(), local_components.size() * count,
                 MPI_INT, 0, world);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_components[0].data(), local_components.size() * count,
                 MPI_INT, 0, world);
  }

  std::vector<Point> h_local;
  for (const auto& component : local_components) {
    auto hull = jarvis(component);
    h_local.insert(h_local.end(), hull.begin(), hull.end());
  }

  if (world.rank() == 0) {
    std::vector<Point> h_merged(c_size);
    int* count_ptr = new int[w_size];
    for (int i = 0; i < w_size; ++i) {
      if (i < rem) {
        count_ptr[i] = count + 1;
      } else {
        count_ptr[i] = count;
      }
    }

    MPI_Gatherv(&h_local[0], h_local.size(), MPI_INT, &h_merged[0], count_ptr, nullptr, MPI_INT, 0, world);
    delete[] count_ptr;

    image = bin_img(jarvis(h_merged), width, height);
  } else {
    MPI_Gatherv(&h_local[0], h_local.size(), MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, world);
  }

  return true;
}

bool ConvexHullBinImgMpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::memcpy(reinterpret_cast<int*>(taskData->outputs[0]), image.data(), image.size() * sizeof(int));
  }

  return true;
}
}  // namespace sidorina_p_convex_hull_binary_image_mpi