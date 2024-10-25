// Filatev Vladislav Sum_of_matrix_elements
#include "seq/filatev_v_sum_of_matrix_elements/include/ops_seq.hpp"

long long filatev_v_sum_of_matrix_elements_seq::sumVector(std::vector<int> vector) {
  long long sum = 0;

  for (long unsigned int i = 0; i < vector.size(); ++i) {
    sum += vector[i];
  }
  return sum;
}

bool filatev_v_sum_of_matrix_elements_seq::SumMatrix::pre_processing() {
  internal_order_test();

  summ = 0;
  size_n = taskData->inputs_count[0];
  size_m = taskData->inputs_count[1];
  matrix = std::vector<int>(size_m * size_n);

  for (int i = 0; i < size_m; ++i) {
    auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

    for (int j = 0; j < size_n; ++j) {
      matrix[i * size_n + j] = temp[j];
    }
  }

  return true;
}

bool filatev_v_sum_of_matrix_elements_seq::SumMatrix::validation() {
  internal_order_test();

  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->outputs_count[0] == 1;
}

bool filatev_v_sum_of_matrix_elements_seq::SumMatrix::run() {
  internal_order_test();

  for (long unsigned int i = 0; i < matrix.size(); ++i) {
    summ += matrix[i];
  }

  return true;
}

bool filatev_v_sum_of_matrix_elements_seq::SumMatrix::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = summ;
  return true;
}
