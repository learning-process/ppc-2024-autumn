#include "seq/smirnov_i_binary_segmentation/include/ops_seq.hpp"

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
    if (img[i] != 0 && img[i] != 255) {
      return false;
    }
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