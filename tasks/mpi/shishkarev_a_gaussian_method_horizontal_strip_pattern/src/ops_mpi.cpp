#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential::pre_processing() {
  internal_order_test();
  // Подготовка входных данных для последовательной обработки
  matrix = *reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
  vector_b = *reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential::validation() {
  internal_order_test();

  // Проверяем наличие taskData и его содержимое
  if (!taskData || taskData->inputs.size() < 2 || taskData->outputs.empty()) {
    return false;  // Недостаточно входных или выходных данных
  }

  // Проверяем корректность матрицы
  auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
  auto* input_vector = reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  auto* output_vector = reinterpret_cast<std::vector<double>*>(taskData->outputs[0]);

  if (input_matrix == nullptr || input_matrix->empty()) {
    return false;  // Матрица отсутствует или пуста
  }

  size_t matrix_size = input_matrix->size();
  for (const auto& row : *input_matrix) {
    if (row.size() != matrix_size) {
      return false;  // Матрица должна быть квадратной
    }
  }

  if (input_vector == nullptr || input_vector->size() != matrix_size) {
    return false;  // Размер вектора должен совпадать с размером матрицы
  }

  if (output_vector == nullptr || output_vector->size() != matrix_size) {
    return false;  // Размер выходного вектора должен совпадать с размером матрицы
  }

  return true;  // Валидация пройдена
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential::run() {
  internal_order_test();

  // Применяем метод Гаусса для последовательной обработки
  size_t n = matrix.size();

  // Приведение матрицы к верхнетреугольному виду
  for (size_t i = 0; i < n; i++) {
    // Нормализация текущей строки
    double pivot = matrix[i][i];
    for (size_t j = i; j < n; j++) {
      matrix[i][j] /= pivot;
    }
    vector_b[i] /= pivot;

    // Обработка всех строк ниже текущей
    for (size_t j = i + 1; j < n; j++) {
      double factor = matrix[j][i];
      for (size_t k = i; k < n; k++) {
        matrix[j][k] -= factor * matrix[i][k];
      }
      vector_b[j] -= factor * vector_b[i];
    }
  }

  // Обратный ход (решение системы уравнений)
  std::vector<double> solution(n, 0);
  for (int i = n - 1; i >= 0; i--) {
    solution[i] = vector_b[i];
    for (size_t j = i + 1; j < n; j++) {
      solution[i] -= matrix[i][j] * solution[j];
    }
  }

  // Сохраняем решение в outputs
  *reinterpret_cast<std::vector<double>*>(taskData->outputs[0]) = solution;
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential::post_processing() {
  internal_order_test();
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Предположим, что taskData содержит матрицу A и вектор b для задачи Гаусса
    matrix = *reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
    vector_b = *reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  }
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel::validation() {
  internal_order_test();

  // Проверка корректности taskData
  if (!taskData || taskData->inputs.size() < 2 || taskData->outputs.empty()) {
    return false;  // Недостаточно входных/выходных данных
  }

  // Проверяем корректность матрицы на нулевом процессе
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
    auto* input_vector = reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
    auto* output_vector = reinterpret_cast<std::vector<double>*>(taskData->outputs[0]);

    if (input_matrix == nullptr || input_matrix->empty()) {
      return false;  // Матрица отсутствует или пуста
    }

    size_t matrix_size = input_matrix->size();
    for (const auto& row : *input_matrix) {
      if (row.size() != matrix_size) {
        return false;  // Матрица должна быть квадратной
      }
    }

    if (input_vector == nullptr || input_vector->size() != matrix_size) {
      return false;  // Размер вектора должен совпадать с размером матрицы
    }

    if (output_vector == nullptr || output_vector->size() != matrix_size) {
      return false;  // Размер выходного вектора должен совпадать с размером матрицы
    }
  }

  // Убеждаемся, что все процессы согласованы
  size_t local_valid = (world.rank() == 0) ? 1 : 0;  // Только процесс 0 выполняет проверки
  size_t global_valid = 0;

  // MPI: Все процессы должны согласовать результат валидации
  boost::mpi::reduce(world, local_valid, global_valid, std::plus<>(), 0);

  // Процесс 0 передаёт результат остальным
  boost::mpi::broadcast(world, global_valid, 0);

  return global_valid > 0;  // Если хотя бы один процесс обнаружил ошибку, метод вернёт false
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel::run() {
  internal_order_test();

  size_t rows_per_process;
  size_t extra_rows;

  if (world.rank() == 0) {
    size_t total_rows = matrix.size();
    rows_per_process = total_rows / world.size();
    extra_rows = total_rows % world.size();
  }

  broadcast(world, rows_per_process, 0);
  broadcast(world, extra_rows, 0);

  std::vector<double> local_matrix_part;
  std::vector<double> local_b_part;

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      size_t start_row = proc * rows_per_process;
      size_t num_rows = (proc == world.size() - 1) ? rows_per_process + extra_rows : rows_per_process;
      world.send(proc, 0, matrix[start_row].data(), num_rows * matrix[start_row].size());
      world.send(proc, 1, vector_b.data() + start_row, num_rows);
    }
    local_matrix_part.assign(matrix[world.rank()].begin(), matrix[world.rank()].end());
    local_b_part.assign(vector_b.begin(), vector_b.end());
  } else {
    size_t num_rows = (world.rank() == world.size() - 1) ? rows_per_process + extra_rows : rows_per_process;
    local_matrix_part.resize(num_rows);
    local_b_part.resize(num_rows);
    world.recv(0, 0, local_matrix_part.data(), num_rows * matrix[0].size());
    world.recv(0, 1, local_b_part.data(), num_rows);
  }

  for (size_t i = 0; i < matrix.size(); i++) {
    for (size_t j = i + 1; j < matrix.size(); j++) {
      if (world.rank() == 0) {
        double factor = matrix[j][i] / matrix[i][i];
        for (size_t k = i; k < matrix[j].size(); k++) {
          matrix[j][k] -= factor * matrix[i][k];
        }
        vector_b[j] -= factor * vector_b[i];
      }
      broadcast(world, matrix[i], 0);
      broadcast(world, vector_b, 0);
    }
  }

  std::vector<double> solution(matrix.size(), 0);
  for (int i = matrix.size() - 1; i >= 0; i--) {
    solution[i] = vector_b[i];
    for (size_t j = i + 1; j < matrix.size(); j++) {
      solution[i] -= matrix[i][j] * solution[j];
    }
    solution[i] /= matrix[i][i];
  }

  boost::mpi::reduce(world, solution, res, std::plus<>(), 0);
  return true;
}

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<double>*>(taskData->outputs[0]) = res;
  }
  return true;
}
