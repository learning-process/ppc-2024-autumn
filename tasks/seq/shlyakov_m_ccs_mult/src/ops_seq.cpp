// Copyright 2024 Nesterov Alexander
// shlyakov_m_min_value_of_row

#include "seq/shlyakov_m_ccs_mult/include/ops_seq.hpp"

using namespace shlyakov_m_ccs_mult;

bool shlyakov_m_ccs_mult::TestTaskSequential::pre_processing() {
  internal_order_test();

  const auto* a_values = reinterpret_cast<const double*>(taskData->inputs[0]);
  const auto* a_row_indices = reinterpret_cast<const int*>(taskData->inputs[1]);
  const auto* a_col_pointers = reinterpret_cast<const int*>(taskData->inputs[2]);

  const auto* b_values = reinterpret_cast<const double*>(taskData->inputs[3]);
  const auto* b_row_indices = reinterpret_cast<const int*>(taskData->inputs[4]);
  const auto* b_col_pointers = reinterpret_cast<const int*>(taskData->inputs[5]);

  A_.values.assign(a_values, a_values + taskData->inputs_count[0]);
  A_.row_indices.assign(a_row_indices, a_row_indices + taskData->inputs_count[1]);
  A_.col_pointers.assign(a_col_pointers, a_col_pointers + taskData->inputs_count[2] + 1);

  B_.values.assign(b_values, b_values + taskData->inputs_count[3]);
  B_.row_indices.assign(b_row_indices, b_row_indices + taskData->inputs_count[4]);
  B_.col_pointers.assign(b_col_pointers, b_col_pointers + taskData->inputs_count[5] + 1);

  rows_a = taskData->inputs_count[2];
  rows_b = taskData->inputs_count[5];
  cols_a = static_cast<int>(A_.col_pointers.size()) - 1;
  cols_b = static_cast<int>(B_.col_pointers.size()) - 1;

  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();

  bool is_invalid = taskData == nullptr || taskData->inputs.size() != 6 || taskData->inputs_count.size() < 6 ||
                    static_cast<int>(taskData->inputs_count[2]) < 0 ||
                    static_cast<int>(taskData->inputs_count[5]) < 0 ||
                    static_cast<int>(taskData->inputs_count[0]) != static_cast<int>(taskData->inputs_count[1]) ||
                    static_cast<int>(taskData->inputs_count[3]) != static_cast<int>(taskData->inputs_count[4]) ||
                    (taskData->inputs_count[0] <= 0 && taskData->inputs_count[3] <= 0);

  return !is_invalid;
}

bool TestTaskSequential::run() {
  internal_order_test();

  result_.col_pointers.clear();
  result_.values.clear();
  result_.row_indices.clear();
  result_.col_pointers.push_back(0);

  std::vector<double> temp;
  temp.resize(rows_a, 0.0);

  int k;
  int pos_a;
  int row_a;
  int b_start;
  int b_end;
  int pos_b;
  double a_val;
  double b_val;

  for (int col_b = 0; col_b < cols_b; ++col_b) {
    std::fill(temp.begin(), temp.end(), 0.0);
    b_start = B_.col_pointers[col_b];
    b_end = B_.col_pointers[col_b + 1];
    for (pos_b = b_start; pos_b < b_end; ++pos_b) {
      b_val = B_.values[pos_b];
      k = B_.row_indices[pos_b];
      int a_start = A_.col_pointers[k];
      int a_end = A_.col_pointers[k + 1];
      for (pos_a = a_start; pos_a < a_end; ++pos_a) {
        a_val = A_.values[pos_a];
        row_a = A_.row_indices[pos_a];
        temp[row_a] += a_val * b_val;
      }
    }
    for (row_a = 0; row_a < rows_a; ++row_a) {
      if (temp[row_a] != 0.0) {
        result_.values.push_back(temp[row_a]);
        result_.row_indices.push_back(row_a);
      }
    }
    result_.col_pointers.push_back(static_cast<int>(result_.values.size()));
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.values.data()));
  taskData->outputs_count.push_back(static_cast<unsigned int>(result_.values.size() * sizeof(double)));

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.row_indices.data()));
  taskData->outputs_count.push_back(static_cast<unsigned int>(result_.row_indices.size() * sizeof(int)));

  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result_.col_pointers.data()));
  taskData->outputs_count.push_back(static_cast<unsigned int>(result_.col_pointers.size() * sizeof(int)));

  return true;
}

// namespace shlyakov_m_ccs_mult
