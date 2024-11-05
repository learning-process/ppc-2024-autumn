#include "seq/gordeva_t_max_val_of_column_matrix/include/ops_seq.hpp"

#include <random>

namespace gordeva_t_max_val_of_column_matrix_seq {

bool gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  int* input_matr;
  input_.resize(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; i++) {
    input_matr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) input_[i][j] = input_matr[j];
  }

  res_.resize(cols);

  return true;
}

bool gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;
  if (taskData->inputs_count.size() < 2) return false;
  if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;

  return true;
}

bool gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < input_[0].size(); i++) {
    int max_el = input_[0][i];
    for (size_t j = 1; j < input_.size(); j++)
      if (input_[j][i] > max_el) max_el = input_[j][i];

    res_[i] = max_el;
  }

  return true;
}

bool gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);

  std::copy(res_.begin(), res_.end(), output_matr);
  return true;
}

std::vector<int> gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::rand_vec(int size, int down, int upp) {
  std::vector<int> v(size);
  for (auto& number : v) number = down + (std::rand() % (upp - down + 1));
  return v;
}

std::vector<std::vector<int>> gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::rand_matr(int rows,
                                                                                                    int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; ++i) {
    matr[i] = rand_vec(cols, -500, 500);
  }
  for (int j = 0; j < cols; ++j) {
    int row_rand = std::rand() % rows;
    matr[row_rand][j] = 10;
  }
  return matr;
}

}  // namespace gordeva_t_max_val_of_column_matrix_seq
