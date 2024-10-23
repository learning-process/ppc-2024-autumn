// Copyright 2024 Nesterov Alexander
#include "seq/kolokolova_d_max_of_vector_elements/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool kolokolova_d_max_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_t row_count = static_cast<size_t>(*taskData->inputs[1]); // Количество строк
  size_t col_count = taskData->inputs_count[0] / row_count; // Количество столбцов

  input_.resize(row_count, std::vector<int>(col_count)); // Инициализация матрицы

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]); // Получаем указатель на входные данные
  for (size_t i = 0; i < row_count; ++i) {
      for (size_t j = 0; j < col_count; ++j) {
          input_[i][j] = input_ptr[i * col_count + j]; // Копируем данные в матрицу
      }
  }

  res.resize(row_count); // Инициализация вектора результата
  return true;
}

bool kolokolova_d_max_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == input_.size();
}

bool kolokolova_d_max_of_vector_elements_seq::TestTaskSequential::run() {
for (size_t i = 0; i < input_.size(); ++i) {
        int max_value = input_[i][0]; // Инициализация первого элемента как максимума

        for (size_t j = 1; j < input_[i].size(); ++j) {
            if (input_[i][j] > max_value) {
                max_value = input_[i][j]; // Обновление максимального значения
            }
        }

        res[i] = max_value; // Запись максимума в результат
    }
    return true;
}

bool kolokolova_d_max_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < res.size(); ++i) {
        output_ptr[i] = res[i]; // Заполнение выходных данных
    }
    return true;
}
