// Copyright 2023 Nesterov Alexander
#include "mpi/poroshin_v_cons_conv_hull_for_bin_image_comp/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>


bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return ((!(taskData->inputs[0] == nullptr) && !(taskData->outputs[0] == nullptr)) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0));
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::run() {
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
      poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::label_connected_components(image);
  std::vector<std::vector<std::pair<int, int>>> coords =
      poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::coordinates_�onnected_�omponents(
          image, count_components);
  for (std::vector<std::pair<int, int>>& t : coords) {
    t = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(t);
  }

  res.clear();

  for (size_t i = 0; i < coords.size(); i++) {
    for (size_t j = 0; j < coords[i].size(); j++) {
      res.push_back(coords[i][j]);
    }
    res.push_back({-1, -1});  // The separating symbol for convex hulls of the connectivity component
  }

  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<std::pair<int, int>*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return ((!(taskData->inputs[0] == nullptr) && !(taskData->outputs[0] == nullptr)) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0));
  }
  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  // Init value for input and output
  int m = 0;
  int n = 0;
  int epsilon = 0;
  int size = 0;
  int local_size = 0;
  int max_rank = 0;

  if (world.rank() == 0) {
    m = taskData->inputs_count[0];
    n = taskData->inputs_count[1];
    size = m * n;

    input_.resize(size);

    for (int i = 0; i < size; i++) {
      input_[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
    }
  }

  boost::mpi::broadcast(world, m, 0);
  boost::mpi::broadcast(world, n, 0);

  // part 1

  if (world.rank() == 0) {
    std::vector<std::vector<int>> image(m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        image[i].push_back(input_[i * n + j]);
      }
    }

    int count_components =
        poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::label_connected_components(image);
    local_input_ =
        poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::coordinates_�onnected_�omponents(
            image, count_components);
  }

  // part 2

  if (world.rank() == 0) {
    size = static_cast<int>(local_input_.size());
  }
  boost::mpi::broadcast(world, size, 0);

  if (world.size() == 1) {
    for (std::vector<std::pair<int, int>>& t : local_input_) {
      t = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(t);
    }

    res.clear();

    for (size_t i = 0; i < local_input_.size(); i++) {
      for (size_t j = 0; j < local_input_[i].size(); j++) {
        res.push_back(local_input_[i][j]);
      }
      res.push_back({-1, -1});  // The separating symbol for convex hulls of the connectivity component
    }

    return true;
  }

  for (int s = 0; s < size; s++) {
    std::vector<std::pair<int, int>> coords;

    if (world.rank() == 0) {
      local_size = static_cast<int>(local_input_[s].size());
    }

    boost::mpi::broadcast(world, local_size, 0);

    if (world.rank() != 0) {
      if (local_size < 5) continue;
    }

    if (world.rank() == 0) {
      std::vector<std::pair<int, int>>& t = local_input_[s];
      int tmp_size = static_cast<int>(t.size());
      if (tmp_size < 5) {
        t = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(t);
        continue;
      }

      if (4 * world.size() >= tmp_size) {
        epsilon = 4;
      } else {
        epsilon = tmp_size / world.size();
      }
      coords.resize(epsilon);
      std::copy(t.begin(), t.begin() + epsilon, coords.begin());
      max_rank = 0;
      int rank = 1;
      for (int i = epsilon; i < tmp_size; i += epsilon) {
        if ((i + epsilon >= tmp_size) || (max_rank == world.size() - 2)) {
          max_rank++;
          world.send(rank, 0, max_rank);
          world.send(rank, 0, tmp_size - epsilon * rank);
          world.send(rank, 0, t.data() + i, tmp_size - epsilon * rank);
          rank++;
          break;
        } else {
          max_rank++;
          world.send(rank, 0, max_rank);
          world.send(rank, 0, epsilon);
          world.send(rank, 0, t.data() + i, epsilon);
          rank++;
        }
      }
      while (rank < world.size()) {
        world.send(rank, 0, max_rank);
        rank++;
      }
    }

    if (world.rank() != 0) {
      world.recv(0, 0, max_rank);
      if (world.rank() <= max_rank) {
        int len_of_data = 0;
        world.recv(0, 0, len_of_data);
        coords.resize(len_of_data);
        world.recv(0, 0, coords.data(), len_of_data);
      }
    }

    coords = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(coords);

    if (world.rank() == 0) {
      int rank = 1;
      while (rank <= max_rank) {
        std::vector<std::pair<int, int>> tmp;
        world.recv(rank, 0, tmp);
        rank++;
        coords.insert(coords.end(), tmp.begin(), tmp.end());
      }

      coords = poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(coords);
      local_input_[s] = coords;

    } else if (world.rank() <= max_rank) {
      world.send(0, 0, coords);
    }
  }

  if (world.rank() == 0) {
    res.clear();

    for (size_t i = 0; i < local_input_.size(); i++) {
      for (size_t j = 0; j < local_input_[i].size(); j++) {
        res.push_back(local_input_[i][j]);
      }
      res.push_back({-1, -1});  // The separating symbol for convex hulls of the connectivity component
    }
  }

  return true;
}

bool poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < res.size(); i++) {
      reinterpret_cast<std::pair<int, int>*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}

std::vector<std::vector<std::pair<int, int>>>
poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::coordinates_�onnected_�omponents(
    std::vector<std::vector<int>>& labeled_image, int count_components) {
  std::vector<std::vector<std::pair<int, int>>> coords(count_components - 1);
  for (int i = 0; i < labeled_image.size(); ++i) {
    for (int j = 0; j < labeled_image[0].size(); ++j) {
      if (labeled_image[i][j] != 0) {
        coords[labeled_image[i][j] - 2].push_back({i, j});
      }
    }
  }

  return coords;
}

int poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::label_connected_components(
    std::vector<std::vector<int>>& image) {
  int label = 2;  // Start with 2 to avoid confusion with pixels 0 and 1
  for (int i = 0; i < image.size(); ++i) {
    for (int j = 0; j < image[0].size(); ++j) {
      if (image[i][j] == 1) {
        std::stack<std::pair<int, int>> pixelStack;
        pixelStack.push({i, j});

        while (!pixelStack.empty()) {
          int x = pixelStack.top().first;
          int y = pixelStack.top().second;
          pixelStack.pop();

          if (x < 0 || x >= image.size() || y < 0 || y >= image[0].size() || image[x][y] != 1) {
            continue;
          }

          image[x][y] = label;

          pixelStack.push({x + 1, y});      // Down
          pixelStack.push({x - 1, y});      // Up
          pixelStack.push({x, y + 1});      // Right
          pixelStack.push({x, y - 1});      // Left
          pixelStack.push({x - 1, y + 1});  // Top left corner
          pixelStack.push({x + 1, y + 1});  // Top right corner
          pixelStack.push({x - 1, y - 1});  // Lower left corner
          pixelStack.push({x + 1, y - 1});  // Lower right corner
        }
        label++;
      }
    }
  }

  return --label;
}

std::vector<std::pair<int, int>> poroshin_v_cons_conv_hull_for_bin_image_comp_mpi::TestMPITaskSequential::convex_hull(
    std::vector<std::pair<int, int>>& inputPoints) {
  auto crossProduct = [](const std::pair<int, int>& origin, const std::pair<int, int>& pointA,
                         const std::pair<int, int>& pointB) {
    return (pointA.first - origin.first) * (pointB.second - origin.second) -
           (pointA.second - origin.second) * (pointB.first - origin.first);
  };

  std::vector<std::pair<int, int>> convexHull;

  if (inputPoints.size() == 0) {
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

  for (size_t i = 1; i < inputPoints.size(); ++i) {
    while (convexHull.size() > 1 &&
           crossProduct(convexHull[convexHull.size() - 2], convexHull[convexHull.size() - 1], inputPoints[i]) <= 0) {
      convexHull.pop_back();
    }
    convexHull.push_back(inputPoints[i]);
  }

  convexHull.push_back(convexHull[0]);
  return convexHull;
}