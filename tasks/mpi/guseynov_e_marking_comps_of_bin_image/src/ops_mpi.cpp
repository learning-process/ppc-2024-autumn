#include "mpi/guseynov_e_marking_comps_of_bin_image/include/ops_mpi.hpp"

#include <map>
#include <random>
#include <vector>

int find(std::map<int, int>& parent, int x) {
  if (parent[x] != x) {
    parent[x] = find(parent, parent[x]);
  }
  return parent[x];
}

void unite(std::map<int, int>& parent, int x, int y) {
  int rootX = find(parent, x);
  int rootY = find(parent, y);
  if (rootX != rootY) {
    parent[rootY] = rootX;
  }
}

void labeling(std::vector<int>& image, std::vector<int>& labeled_image, int rows, int columns) {
  std::vector<int> label_equivalence;
  int current_label = 2;
  std::map<int, int> parent;
  // Displacements for neighbours
  int dx[] = {-1, 1, 0, 0, -1, 1};
  int dy[] = {0, 0, -1, 1, 1, -1};

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int position = x * columns + y;
      if (image_[position] == 0) {
        std::vector<int> neighbours;

        for (int i = 0; i < 6; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int tmp_pos = nx * columns + ny;
          if (nx >= 0 && nx < rows && ny >= 0 && ny < columns && (labeled_image[tmp_pos] > 1)) {
            neighbours.push_back(labeled_image[tmp_pos]);
          }
        }

        if (neighbours.empty()) {
          labeled_image[position] = current_label;
          parent[current_label] = current_label;
          current_label++;
        } else {
          int min_label = *min_element(neighbours.begin(), neighbours.end());
          labeled_image[position] = min_label;

          for (int label : neighbours) {
            unite(parent, min_label, label);
          }
        }
      }
    }
  }

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int position = x * columns + y;
      if (labeled_image[position] > 1) {
        labeled_image[position] = find(parent, labeled_image[position]);
      }
    }
  }
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];
  int pixels_count = rows * columns;
  image_ = std::vector<int>(pixels_count);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + pixels_count, image_.begin());

  labeled_image = std::vector<int>(rows * columns, 1);
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  int tmp_rows = taskData->inputs_count[0];
  int tmp_columns = taskData->inputs_count[1];

  for (int x = 0; x < tmp_rows; x++) {
    for (int y = 0; y < tmp_columns; y++) {
      int pixel = static_cast<int>(taskData->inputs[0][x * tmp_columns + y]);
      if (pixel < 0 || pixel > 1) {
        return false;
      }
    }
  }
  return tmp_rows > 0 && tmp_columns > 0 && static_cast<int>(taskData->outputs_count[0]) == tmp_rows &&
         static_cast<int>(taskData->outputs_count[1]) == tmp_columns;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  labeling(image_, labeled_image, rows, columns);
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(labeled_image.begin(), labeled_image.end(), outputPtr);
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
    int pixels_count = rows * columns;
    image_ = std::vector<int>(pixels_count);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + pixels_count, image_.begin());

    labeled_image = std::vector<int>(rows * columns, 1);
  }
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    int tmp_rows = taskData->inputs_count[0];
    int tmp_columns = taskData->inputs_count[1];

    for (int x = 0; x < tmp_rows; x++) {
      for (int y = 0; y < tmp_columns; y++) {
        int pixel = static_cast<int>(taskData->inputs[0][x * tmp_columns + y]);
        if (pixel < 0 || pixel > 1) {
          return false;
        }
      }
    }
    return tmp_rows > 0 && tmp_columns > 0 && static_cast<int>(taskData->outputs_count[0]) == tmp_rows &&
           static_cast<int>(taskData->outputs_count[1]) == tmp_columns;
  }
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  std::vector<int> row_counts(world.size(), rows / world.size()) for (int i = 0; i < rows % world.size(); i++) {
    row_counts[i]++;
  }

  int local_rows = row_counts[world.rank()];
  local_image_ = std::vector<int>(columns * local_rows);
  boost::mpi::scatterv(world, image_, row_counts, local_image_, 0);

  std::vector<int> local_labeled_image(columns * local_rows, 1);
  labeling(local_image_, local_labeled_image, local_rows, columns);
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(labeled_image.begin(), labeled_image.end(), outputPtr);
  }
  return true;
}