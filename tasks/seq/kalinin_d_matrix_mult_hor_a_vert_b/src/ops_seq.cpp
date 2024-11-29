#include "seq/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool kalinin_d_matrix_mult_hor_a_vert_b::MultHorAVertBTaskSequential::pre_processing() {
  internal_order_test();

  // Считываем указатели на входные данные
  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  // Читаем размеры матриц
  rows_A = taskData->inputs_count[0];     // Число строк A
  columns_A = taskData->inputs_count[1];  // Число столбцов A
  rows_B = taskData->inputs_count[2];     // Число строк B
  columns_B = taskData->inputs_count[3];  // Число столбцов B

  // Проверяем совместимость матриц
  if (columns_A != rows_B) {
    return false;  // Размеры матриц не совместимы для умножения
  }

  // Инициализируем результат
  C.assign(rows_A * columns_B, 0);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::MultHorAVertBTaskSequential::validation() {
  internal_order_test();
  
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[2] > 0 && taskData->inputs_count[3] > 0 &&
         taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool kalinin_d_matrix_mult_hor_a_vert_b::MultHorAVertBTaskSequential::run() {
  internal_order_test();

  // Цикл умножения
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < columns_B; ++j) {
      for (int k = 0; k < columns_A; ++k) {
        // Обновляем результат
        C[i * columns_B + j] += input_A[i * columns_A + k] * input_B[k * columns_B + j];
      }
    }
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::MultHorAVertBTaskSequential::post_processing() {
  internal_order_test();

  // Размер результирующей матрицы
  size_t total_size = rows_A * columns_B;

  // Копируем результат в выходной буфер
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

  for (size_t i = 0; i < total_size; ++i) {
    output_data[i] = C[i];
  }

  return true;
}
