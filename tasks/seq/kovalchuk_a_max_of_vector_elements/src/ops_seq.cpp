// Copyright 2023 Nesterov Alexander
#include <climits>
#include <limits>
#include <numeric>
#include <random>
<<<<<<< HEAD
#include <algorithm>
=======
>>>>>>> d8cdde3175d86a868290c246bfeb882051127316

#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

namespace kovalchuk_a_max_of_vector_elements_seq {

std::vector<int> getRandomVector(int size, int start_gen, int fin_gen) {
  static std::random_device dev;
  static std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(start_gen, fin_gen);
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int start_gen, int fin_gen) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; i++) {
    matrix[i] = getRandomVector(columns, start_gen, fin_gen);
  }
  return matrix;
}

bool TestTaskSequential::pre_processing() {
  inputMatrix_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      inputMatrix_[i][j] = tmp_ptr[j];
    }
  }
  result_ = std::numeric_limits<int>::min();
  return true;
}

bool TestTaskSequential::validation() {
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool TestTaskSequential::run() {
  result_ = std::numeric_limits<int>::min();
  for (const auto& row : inputMatrix_) {
    int row_max = *std::max_element(row.begin(), row.end());
    if (row_max > result_) {
      result_ = row_max;
    }
  }
  return true;
}

bool TestTaskSequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  return true;
}

}  // namespace kovalchuk_a_max_of_vector_elements_seq