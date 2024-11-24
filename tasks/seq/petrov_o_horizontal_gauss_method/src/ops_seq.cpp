#include "seq/petrov_o_horizontal_gauss_method/include/ops_seq.hpp"

#include <vector>
#include <algorithm>


bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::validation() {
    internal_order_test();

        // Проверка количества входных/выходных данных
    if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
        return false;
    }

    size_t n = taskData->inputs_count[0];

    if (n == 0 ) { 
        return false;
    }


    // Проверка наличия указателей на данные
    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
        return false;
    }


    return true;
}

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::pre_processing() {
  internal_order_test();

  // Получаем n из первого входного параметра
  size_t n = taskData->inputs_count[0];

  matrix.resize(n, std::vector<double>(n));
  b.resize(n);
  x.resize(n);


  double* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
  for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j < n; ++j) {
          matrix[i][j] = matrix_input[i * n + j];
      }
  }

  double* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
  for(size_t i = 0; i < n; ++i) {
      b[i] = b_input[i];
  }


  return true;
}

bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::run() {
    internal_order_test();

    size_t n = matrix.size();

    // Прямой ход
    for (size_t k = 0; k < n - 1; ++k) {
        
        // Проверка на вырожденность
        if (abs(matrix[k][k]) < 1e-10) {
            return false; // Матрица вырождена
        }

        for (size_t i = k + 1; i < n; ++i) {
            double factor = matrix[i][k] / matrix[k][k];
            for (size_t j = k; j < n; ++j) { // Исправлено: начинаем с k
                matrix[i][j] -= factor * matrix[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Обратный ход
    x[n - 1] = b[n - 1] / matrix[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        double sum = b[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= matrix[i][j] * x[j];
        }
        x[i] = sum / matrix[i][i];
    }

    return true;
}


bool petrov_o_horizontal_gauss_method_seq::GaussHorizontalSequential::post_processing() {
  internal_order_test();

  double* output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < x.size(); ++i) {
        output[i] = x[i];
    }
  return true;
}