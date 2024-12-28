#include "mpi/smirnov_i_binary_segmentation/include/ops_mpi.hpp"

#include <thread>

using namespace std::chrono_literals;

bool smirnov_i_binary_segmentation::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  mask = std::vector<int>(cols * rows, 1);
  for (size_t i = 0; i < img.size(); i++) {
    if (img[i] == 255) {
      img[i] = 1;
    }
  }
  return true;
}

bool smirnov_i_binary_segmentation::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() != 2) return false;
  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];
  if (rows < 0 || cols < 0) return false;
  img = std::vector<int>(cols * rows, 1);
  auto* tmp_ptr_img = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr_img, tmp_ptr_img + (cols * rows), img.begin());

  // check is binary
  for (size_t i = 0; i < img.size(); i++) {
    if (img[i] != 0 && img[i] != 255) return false;
  }
  return true;
}
std::vector<int> smirnov_i_binary_segmentation::TestMPITaskSequential::make_border(const std::vector<int>& img_,
                                                                                   int cols_, int rows_) {
  std::vector<int> new_img((cols_ + 2) * (rows_ + 2), 1);

  for (int row = 0; row < rows_; row++) {
    int src_start = row * cols_;
    int dst_start = (row + 1) * (cols_ + 2) + 1;
    std::copy(&img_[src_start], &img_[src_start + cols_], &new_img[dst_start]);
  }

  return new_img;
}

std::vector<int> smirnov_i_binary_segmentation::TestMPITaskSequential::del_border(const std::vector<int>& img_,
                                                                                  int cols_, int rows_) {
  int bordered_cols = cols_ + 2;
  std::vector<int> result(cols_ * rows_);

  for (int row = 0; row < rows_; row++) {
    int src_start = (row + 1) * bordered_cols + 1;
    int dst_start = row * cols_;

    std::copy(&img_[src_start], &img_[src_start + cols_], &result[dst_start]);
  }

  return result;
}

void smirnov_i_binary_segmentation::TestMPITaskSequential::merge_equivalence(std::map<int, std::set<int>>& eq_table,
                                                                             int a, int b) {
  if (a == b) return;

  auto find_a = eq_table.find(a);
  auto find_b = eq_table.find(b);

  if (find_a == eq_table.end() && find_b == eq_table.end()) {
    eq_table[a] = {b};
    eq_table[b] = {a};
  } else if (find_a != eq_table.end() && find_b == eq_table.end()) {
    eq_table[a].insert(b);
    eq_table[b] = eq_table[a];
  } else if (find_a == eq_table.end() && find_b != eq_table.end()) {
    eq_table[b].insert(a);
    eq_table[a] = eq_table[b];
  } else {
    eq_table[a].insert(eq_table[b].begin(), eq_table[b].end());
    eq_table[b] = eq_table[a];
  }
}
bool smirnov_i_binary_segmentation::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> bord_img = make_border(img, cols, rows);
  std::map<int, std::set<int>> eq_table;
  int bord_cols = cols + 2;
  int px_A;
  int px_B;
  int px_C;
  int px_D;

  int mark = 2;
  // D B
  // C A

  for (size_t i = bord_cols + 1; i < bord_img.size() - bord_cols; i++) {
    if (i % bord_cols == 0 || static_cast<int>(i) % bord_cols == bord_cols - 1) {
      continue;
    }
    px_A = i;
    px_B = i - bord_cols;
    px_C = i - 1;
    px_D = i - 1 - bord_cols;
    if (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] == 1 && bord_img[px_D] == 1) {
      bord_img[px_A] = mark;
      mark++;
    } else if (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] == 1 && bord_img[px_D] != 1) {
      bord_img[px_A] = bord_img[px_D];
    } else if ((bord_img[px_A] == 0 && bord_img[px_C] != 1 && bord_img[px_B] == 1) ||
               (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] != 1)) {
      bord_img[px_A] = std::max(bord_img[px_C], bord_img[px_B]);
    } else if (bord_img[px_A] == 0 && bord_img[px_C] != 1 && bord_img[px_B] != 1) {
      if (bord_img[px_C] == bord_img[px_B]) {
        bord_img[px_A] = bord_img[px_B];
      } else {
        bord_img[px_A] = std::min(bord_img[px_C], bord_img[px_B]);
        merge_equivalence(eq_table, bord_img[px_B], bord_img[px_C]);
      }
    }
  }

  for (auto& pair : eq_table) {
    pair.second.insert(pair.first);
  }

  for (auto& pair : eq_table) {
    std::set<int>& values = pair.second;

    for (int v : values) {
      eq_table[v].insert(values.begin(), values.end());
    }
  }

  for (size_t i = bord_cols + 1; i < (bord_img.size() - bord_cols); i++) {
    if (i % bord_cols == 0 || static_cast<int>(i) % bord_cols == bord_cols - 1) {
      continue;
    }

    int label = bord_img[i];
    auto find_label = eq_table.find(label);
    if (find_label != eq_table.end()) {
      bord_img[i] = *std::min_element(find_label->second.begin(), find_label->second.end());
    }
  }
  mask = del_border(bord_img, cols, rows);
  // rename marks
  for (int i = 1; i < cols * rows; i++) {
    if (mask[i - 1] != mask[i] && mask[i - 1] != 1 && mask[i] != 1 && i % cols != 0) {
      merge_equivalence(eq_table, mask[i - 1], mask[i]);
    }
  }
  for (int i = cols; i < cols * rows; i++) {
    if (mask[i - cols] != mask[i] && mask[i - cols] != 1 && mask[i] != 1) {
      merge_equivalence(eq_table, mask[i - cols], mask[i]);
    }
  }
  for (int i = cols + 1; i < cols * rows; i++) {
    if (mask[i - cols - 1] != mask[i] && mask[i - cols - 1] != 1 && mask[i] != 1 && i % cols != 0) {
      merge_equivalence(eq_table, mask[i - cols - 1], mask[i]);
    }
  }

  for (auto& pair : eq_table) {
    std::set<int>& values = pair.second;

    for (int v : values) {
      eq_table[v].insert(values.begin(), values.end());
    }
  }

  for (size_t i = 0; i < (mask.size()); i++) {
    int label = mask[i];
    auto find_label = eq_table.find(label);
    if (find_label != eq_table.end()) {
      mask[i] = *std::min_element(find_label->second.begin(), find_label->second.end());
    }
  }

  std::set<int> unique_labels(mask.begin(), mask.end());
  unique_labels.erase(0);
  unique_labels.erase(1);

  std::map<int, int> label_mapping;
  int new_label = 2;

  for (int label : unique_labels) {
    auto it = eq_table.find(label);
    if (it != eq_table.end()) {
      int canonical_label = *std::min_element(it->second.begin(), it->second.end());
      if (label_mapping.find(canonical_label) == label_mapping.end()) {
        label_mapping[canonical_label] = new_label++;
      }
      label_mapping[label] = label_mapping[canonical_label];
    } else {
      label_mapping[label] = new_label++;
    }
  }
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] > 1) {
      mask[i] = label_mapping[mask[i]];
    }
  }
  return true;
}

bool smirnov_i_binary_segmentation::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(mask.data(), mask.data() + cols * rows, tmp_ptr);
  return true;
}

bool smirnov_i_binary_segmentation::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    mask = std::vector<int>(cols * rows, 1);
    for (size_t i = 0; i < img.size(); i++) {
      if (img[i] == 255) {
        img[i] = 1;
      }
    }
  }
  return true;
}

bool smirnov_i_binary_segmentation::TestMPITaskParallel::validation() {
  internal_order_test();
  bool is_valid = true;
  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 2) is_valid = false;
    if (is_valid) {
      rows = taskData->inputs_count[0];
      cols = taskData->inputs_count[1];
      if (rows < 0 || cols < 0) is_valid = false;
      if (is_valid) {
        img = std::vector<int>(cols * rows, 1);
        auto* tmp_ptr_img = reinterpret_cast<int*>(taskData->inputs[0]);
        std::copy(tmp_ptr_img, tmp_ptr_img + (cols * rows), img.begin());
        for (size_t i = 0; i < img.size(); i++) {
          if (img[i] != 0 && img[i] != 255) {
            is_valid = false;
            break;
          }
        }
      }
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

  return is_valid;
}
std::vector<int> smirnov_i_binary_segmentation::TestMPITaskParallel::make_border(const std::vector<int>& img_,
                                                                                 int cols_, int rows_) {
  std::vector<int> new_img((cols_ + 2) * (rows_ + 2), 1);
  for (int row = 0; row < rows_; row++) {
    int src_start = row * cols_;
    int dst_start = (row + 1) * (cols_ + 2) + 1;
    std::copy(&img_[src_start], &img_[src_start + cols_], &new_img[dst_start]);
  }
  return new_img;
}
std::vector<int> smirnov_i_binary_segmentation::TestMPITaskParallel::del_border(const std::vector<int>& img_, int cols_,
                                                                                int rows_) {
  int bordered_cols = cols_ + 2;
  std::vector<int> result(cols_ * rows_);
  for (int row = 0; row < rows_; row++) {
    int src_start = (row + 1) * bordered_cols + 1;
    int dst_start = row * cols_;

    std::copy(&img_[src_start], &img_[src_start + cols_], &result[dst_start]);
  }

  return result;
}

void smirnov_i_binary_segmentation::TestMPITaskParallel::merge_equivalence(std::map<int, std::set<int>>& eq_table,
                                                                           int a, int b) {
  if (a == b) return;

  auto find_a = eq_table.find(a);
  auto find_b = eq_table.find(b);

  if (find_a == eq_table.end() && find_b == eq_table.end()) {
    eq_table[a] = {b};
    eq_table[b] = {a};
  } else if (find_a != eq_table.end() && find_b == eq_table.end()) {
    eq_table[a].insert(b);
    eq_table[b] = eq_table[a];
  } else if (find_a == eq_table.end() && find_b != eq_table.end()) {
    eq_table[b].insert(a);
    eq_table[a] = eq_table[b];
  } else {
    eq_table[a].insert(eq_table[b].begin(), eq_table[b].end());
    eq_table[b] = eq_table[a];
  }
}
std::string smirnov_i_binary_segmentation::TestMPITaskParallel::serialize_eq_table(
    const std::map<int, std::set<int>>& eq_table) {
  std::ostringstream oss;
  for (const auto& pair : eq_table) {
    int key = pair.first;
    auto values = pair.second;
    oss << key << ":";
    for (int value : values) {
      oss << value << ",";
    }
    oss << ";";
  }
  return oss.str();
}
std::map<int, std::set<int>> smirnov_i_binary_segmentation::TestMPITaskParallel::deserialize_eq_table(
    const std::string& str) {
  std::map<int, std::set<int>> eq_table;
  std::istringstream iss(str);
  std::string entry;

  while (std::getline(iss, entry, ';')) {
    if (entry.empty()) continue;

    size_t colon_pos = entry.find(':');
    int key = std::stoi(entry.substr(0, colon_pos));
    std::set<int> values;

    std::string values_str = entry.substr(colon_pos + 1);
    std::istringstream value_stream(values_str);
    std::string value;

    while (std::getline(value_stream, value, ',')) {
      if (!value.empty()) {
        values.insert(std::stoi(value));
      }
    }

    eq_table[key] = values;
  }
  return eq_table;
}
void smirnov_i_binary_segmentation::TestMPITaskParallel::merge_global_eq_table(
    std::map<int, std::set<int>>& global_eq_table, const std::map<int, std::set<int>>& local_eq_table) {
  for (const auto& pair : local_eq_table) {
    int key = pair.first;
    auto values = pair.second;

    if (global_eq_table.find(key) == global_eq_table.end()) {
      global_eq_table[key] = values;
    } else {
      global_eq_table[key].insert(values.begin(), values.end());
    }

    for (int value : values) {
      global_eq_table[value].insert(key);
    }
  }
}
bool smirnov_i_binary_segmentation::TestMPITaskParallel::run() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[2];
  if (rank == 0) {
    dims[0] = rows;
    dims[1] = cols;
  }

  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

  rows = dims[0];
  cols = dims[1];

  if (rank != 0) {
    img.resize(cols * rows);
  }

  MPI_Bcast(img.data(), cols * rows, MPI_INT, 0, MPI_COMM_WORLD);
  int* sendcounts = new int[size];
  std::fill(sendcounts, sendcounts + size, 0);
  int* displs = new int[size]();
  int rows_per_proc = rows / size;
  int extra_rows = rows % size;
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      sendcounts[i] = (rows_per_proc + 1) * cols;
    } else {
      sendcounts[i] = rows_per_proc * cols;
    }
    displs[i] = offset;
    offset += sendcounts[i];
  }
  std::vector<int> local_img(sendcounts[rank], 1);
  MPI_Scatterv(img.data(), sendcounts, displs, MPI_INT, local_img.data(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> bord_img = make_border(local_img, cols, sendcounts[rank] / cols);
  std::map<int, std::set<int>> local_eq_table;
  int bord_cols = cols + 2;
  int px_A;
  int px_B;
  int px_C;
  int px_D;
  int mark = 2 + rank * (sendcounts[0]) * 2;
  // D B
  // C A
  // direct local pass
  for (size_t i = bord_cols + 1; i < bord_img.size() - bord_cols; i++) {
    if (i % bord_cols == 0 || static_cast<int>(i) % bord_cols == bord_cols - 1) {
      continue;
    }
    px_A = i;
    px_B = i - bord_cols;
    px_C = i - 1;
    px_D = i - 1 - bord_cols;
    if (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] == 1 && bord_img[px_D] == 1) {
      bord_img[px_A] = mark;
      mark++;
    } else if (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] == 1 && bord_img[px_D] != 1) {
      bord_img[px_A] = bord_img[px_D];
    } else if ((bord_img[px_A] == 0 && bord_img[px_C] != 1 && bord_img[px_B] == 1) ||
               (bord_img[px_A] == 0 && bord_img[px_C] == 1 && bord_img[px_B] != 1)) {
      bord_img[px_A] = std::max(bord_img[px_C], bord_img[px_B]);
    } else if (bord_img[px_A] == 0 && bord_img[px_C] != 1 && bord_img[px_B] != 1) {
      if (bord_img[px_C] == bord_img[px_B]) {
        bord_img[px_A] = bord_img[px_B];
      } else {
        bord_img[px_A] = std::min(bord_img[px_C], bord_img[px_B]);
        merge_equivalence(local_eq_table, bord_img[px_B], bord_img[px_C]);
      }
    }
  }

  for (auto& pair : local_eq_table) {
    std::set<int>& values = pair.second;
    for (int v : values) {
      local_eq_table[v].insert(values.begin(), values.end());
    }
  }

  for (size_t i = bord_cols + 1; i < (bord_img.size() - bord_cols); i++) {
    if (i % bord_cols == 0 || static_cast<int>(i) % bord_cols == bord_cols - 1) {
      continue;
    }

    int label = bord_img[i];
    auto find_label = local_eq_table.find(label);
    if (find_label != local_eq_table.end()) {
      bord_img[i] = *std::min_element(find_label->second.begin(), find_label->second.end());
    }
  }

  std::vector<int> local_mask;
  local_mask = del_border(bord_img, cols, sendcounts[rank] / cols);
  if (rank != 0) {
    mask = std::vector<int>();
  }
  int* recvcounts = new int[size]();
  int* recvdispls = new int[size]();
  offset = 0;

  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      recvcounts[i] = (rows_per_proc + 1) * cols;
    } else {
      recvcounts[i] = rows_per_proc * cols;
    }
    recvdispls[i] = offset;
    offset += recvcounts[i];
  };
  MPI_Gatherv(local_mask.data(), sendcounts[rank], MPI_INT, mask.data(), recvcounts, recvdispls, MPI_INT, 0,
              MPI_COMM_WORLD);

  std::map<int, std::set<int>> global_eq_table;

  std::string local_eq_table_str = serialize_eq_table(local_eq_table);
  int str_len = local_eq_table_str.size();
  std::vector<int> lengths(size);
  MPI_Gather(&str_len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> recv_buf(std::accumulate(lengths.begin(), lengths.end(), 0));
  std::vector<int> offsets(size, 0);
  std::partial_sum(lengths.begin(), lengths.end() - 1, offsets.begin() + 1);
  MPI_Gatherv(local_eq_table_str.data(), str_len, MPI_CHAR, recv_buf.data(), lengths.data(), offsets.data(), MPI_CHAR,
              0, MPI_COMM_WORLD);
  // resolving border conflicts and renaming marks
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      std::map<int, std::set<int>> temp_eq_table = deserialize_eq_table(std::string(&recv_buf[offsets[i]], lengths[i]));
      merge_global_eq_table(global_eq_table, temp_eq_table);
    }

    for (int i = 1; i < cols * rows; i++) {
      if (mask[i - 1] != mask[i] && mask[i - 1] != 1 && mask[i] != 1 && i % cols != 0) {
        merge_equivalence(global_eq_table, mask[i - 1], mask[i]);
      }
    }
    for (int i = cols; i < cols * rows; i++) {
      if (mask[i - cols] != mask[i] && mask[i - cols] != 1 && mask[i] != 1) {
        merge_equivalence(global_eq_table, mask[i - cols], mask[i]);
      }
    }
    for (int i = cols + 1; i < cols * rows; i++) {
      if (mask[i - cols - 1] != mask[i] && mask[i - cols - 1] != 1 && mask[i] != 1 && i % cols != 0) {
        merge_equivalence(global_eq_table, mask[i - cols - 1], mask[i]);
      }
    }
    for (auto& pair : global_eq_table) {
      std::set<int>& values = pair.second;

      for (int v : values) {
        global_eq_table[v].insert(values.begin(), values.end());
      }
    }

    for (size_t i = 0; i < (mask.size()); i++) {
      int label = mask[i];
      auto find_label = global_eq_table.find(label);
      if (find_label != global_eq_table.end()) {
        mask[i] = *std::min_element(find_label->second.begin(), find_label->second.end());
      }
    }
    std::set<int> unique_labels(mask.begin(), mask.end());
    unique_labels.erase(0);
    unique_labels.erase(1);

    std::map<int, int> label_mapping;

    int new_label = 2;

    for (int label : unique_labels) {
      auto it = global_eq_table.find(label);
      if (it != global_eq_table.end()) {
        int canonical_label = *std::min_element(it->second.begin(), it->second.end());
        if (label_mapping.find(canonical_label) == label_mapping.end()) {
          label_mapping[canonical_label] = new_label++;
        }
        label_mapping[label] = label_mapping[canonical_label];
      } else {
        label_mapping[label] = new_label++;
      }
    }
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i] > 1) {
        mask[i] = label_mapping[mask[i]];
      }
    }
  }

  return true;
}

bool smirnov_i_binary_segmentation::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cols * rows; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = mask[i];
    }
  }
  return true;
}