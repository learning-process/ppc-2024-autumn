#include "mpi/guseynov_e_marking_comps_of_bin_image/include/ops_mpi.hpp"

#include <map>
#include <vector>
#include <queue>
#include <set>

int findParent(std::map<int, std::set<int>>& parent, int labl) {
    auto srch = parent.find(labl);
    if (srch != parent.end()){
      return *srch->second.begin();
    }

    return labl;
}

void fixTable(std::map<int, std::set<int>>& parent){
  for (auto& pair : parent){
    for (auto value : pair.second){
        parent[value].insert(pair.second.begin(), pair.second.end());
    }
  }
}

void fixLabels(std::vector<int>& labeled_image, int rows, int cols){
  std::map<int, int> labels_equivalence;
  int min_label = 2;
  for (int x = 0; x < rows; x++){
    for (int y = 0; y < cols; y++){
      int position = x * cols + y;
      if (labeled_image[position] > 1){
        int final_label;
        auto srch_label = labels_equivalence.find(labeled_image[position]);
        if (srch_label == labels_equivalence.end()){
          final_label = min_label;
          labels_equivalence[labeled_image[position]] = min_label++;
        }else{
          final_label = srch_label->second;
        }

        labeled_image[position] = final_label;
      }
    }
  }
} 


void unite(std::map<int, std::set<int>>& parent, int new_label, int neighbour_label) {
  if (new_label == neighbour_label){
    return;
  }

  auto srch1 = parent.find(new_label);
  auto srch2 = parent.find(neighbour_label);

  if (srch1 == parent.end() && srch2 == parent.end()){
    parent[new_label].insert(neighbour_label);
    parent[new_label].insert(new_label);
    parent[neighbour_label].insert(new_label);
    parent[neighbour_label].insert(neighbour_label);
  }else if(srch1 != parent.end() && srch2 == parent.end()){
    parent[new_label].insert(neighbour_label);
    parent[neighbour_label] = parent[new_label];
  }else if(srch1 == parent.end() && srch2 != parent.end()){
    parent[neighbour_label].insert(new_label);
    parent[new_label] = parent[neighbour_label];
  }else{
    std::set<int> tmp_set = parent[new_label];
    parent[new_label].insert(parent[neighbour_label].begin(), parent[neighbour_label].end());
    parent[neighbour_label].insert(parent[new_label].begin(), parent[new_label].end());
  }
}

void labeling(std::vector<int>& image, std::vector<int>& labeled_image, int rows, int columns, int min_label) {
  
  int current_label = 2;
  std::map<int, std::set<int>> parent;
  // Displacements for neighbours
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int position = x * columns + y;
      if (image[position] == 0) {
        std::vector<int> neighbours;

        for (int i = 0; i < 3; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int tmp_pos = nx * columns + ny;
          if (nx >= 0 && nx < rows && ny >= 0 && ny < columns && (labeled_image[tmp_pos] > 1)) {
            neighbours.push_back(labeled_image[tmp_pos]);
          }
        }

        if (neighbours.empty()) {
          labeled_image[position] = current_label;
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
  fixTable(parent);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int position = x * columns + y;
      if (labeled_image[position] > 1) {
        int find_label = findParent(parent, labeled_image[position]);
        
        labeled_image[position] = find_label;       
        
        }
      }
    }
  fixLabels(labeled_image, rows, columns);
}


void labelingFixForMPI(std::vector<int>& labeled_image, int rows, int columns) {
  int current_label = 2;
  int dx[] = {-1, 1, 0, 0, -1, 1};
  int dy[] = {0, 0, -1, 1, 1, -1};

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int position = x * columns + y;
      if (labeled_image[position] > 1) {
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
          current_label++;
        } else {
          int min_label = *min_element(neighbours.begin(), neighbours.end());
          if (current_label <= min_label) {
            min_label = current_label;
            current_label++;
          }
          labeled_image[position] = min_label;
           for (int i = 0; i < 6; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int tmp_pos = nx * columns + ny;
          if (nx >= 0 && nx < rows && ny >= 0 && ny < columns && (labeled_image[tmp_pos] > 1)) {
            labeled_image[tmp_pos] = min_label;
          }
          }
        }
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

  labeling(image_, labeled_image, rows, columns, 2);
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

  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, columns, 0);

  std::vector<int> sizes(world.size(), rows / world.size() * columns);
  for (int i = 0; i < rows % world.size(); i++) {
    sizes[i] += columns;
  }

  local_image_ = std::vector<int>(sizes[world.rank()]);
  boost::mpi::scatterv(world, image_, sizes, local_image_.data(), 0);

  std::vector<int> local_labeled_image(sizes[world.rank()], 1);
  int min_label = world.rank() * sizes[world.rank()] / 2 + 2;
  labeling(local_image_, local_labeled_image, sizes[world.rank()] / columns, columns, min_label);

  boost::mpi::gatherv(world, local_labeled_image, labeled_image.data(), sizes, 0);

  if (world.rank() == 0) {
    labelingFixForMPI(labeled_image, rows, columns);
  }
  return true;
}

bool guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(labeled_image.begin(), labeled_image.end(), outputPtr);
  }
  return true;
}