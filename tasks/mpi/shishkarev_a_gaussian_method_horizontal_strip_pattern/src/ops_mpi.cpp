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

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential::validation() {
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

bool shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalSequential::validation() {
  internal_order_test();

  // Проверка наличия taskData и корректности входных/выходных данных
  if (!taskData || taskData->inputs.size() < 2 || taskData->outputs.empty()) {
    return false;  // Недостаточно входных или выходных данных
  }

  // Проверка корректности данных
  auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
  auto* input_vector = reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
  auto* output_vector = reinterpret_cast<std::vector<double>*>(taskData->outputs[0]);

  if (input_matrix == nullptr || input_matrix->empty()) {
    return false;  // Матрица отсутствует или пуста
  }

  size_t matrix_size = input_matrix->size();  // Количество строк матрицы (вектор векторов)

  // Проверка, что все строки матрицы имеют одинаковую длину (матрица не рваная)
  for (const auto& row : *input_matrix) {
    if (row.size() != matrix_size) {
      return false;  // Матрица не квадратная, т.е. строки имеют разную длину
    }
  }

  // Проверка размера вектора правой части
  if (input_vector == nullptr || input_vector->size() != matrix_size) {
    return false;  // Размер вектора должен совпадать с размером матрицы
  }

  // Проверка размера выходного вектора
  if (output_vector == nullptr || output_vector->size() != matrix_size) {
    return false;  // Размер выходного вектора должен совпадать с размером матрицы
  }

  // Проверка диагональных элементов матрицы
  for (size_t i = 0; i < matrix_size; ++i) {
    if ((*input_matrix)[i][i] == 0.0) {
      std::cout << "Warning: Zero diagonal element at position (" << i << ", " << i << ")" << std::endl;
      return false;  // Если диагональный элемент равен нулю, то валидация не пройдена
    }
  }

  return true;  // Валидация пройдена успешно
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

  // Проверка наличия taskData и корректности входных/выходных данных
  if (!taskData || taskData->inputs.size() < 2 || taskData->outputs.empty()) {
    return false;  // Недостаточно входных или выходных данных
  }

  // Проверка корректности данных на процессе с рангом 0
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
    auto* input_vector = reinterpret_cast<std::vector<double>*>(taskData->inputs[1]);
    auto* output_vector = reinterpret_cast<std::vector<double>*>(taskData->outputs[0]);

    if (input_matrix == nullptr || input_matrix->empty()) {
      return false;  // Матрица отсутствует или пуста
    }

    size_t matrix_size = input_matrix->size();  // Количество строк матрицы (вектор векторов)

    // Проверка, что все строки матрицы имеют одинаковую длину (матрица не рваная)
    for (const auto& row : *input_matrix) {
      if (row.size() != matrix_size) {
        return false;  // Матрица не квадратная, т.е. строки имеют разную длину
      }
    }

    if (input_vector == nullptr || input_vector->size() != matrix_size) {
      return false;  // Размер вектора должен совпадать с размером матрицы
    }

    if (output_vector == nullptr || output_vector->size() != matrix_size) {
      return false;  // Размер выходного вектора должен совпадать с размером матрицы
    }

    // Проверка диагональных элементов матрицы
    for (size_t i = 0; i < matrix_size; ++i) {
      if ((*input_matrix)[i][i] == 0.0) {
        std::cout << "Warning: Zero diagonal element at position (" << i << ", " << i << ")" << std::endl;
        return false;  // Если диагональный элемент равен нулю, то валидация не пройдена
      }
    }
  }

  // Шаг 1: Проверка корректности локальных данных на каждом процессе
  size_t local_valid = 1;  // Предполагаем, что данные локально валидны

  // Каждый процесс проверяет локальную часть данных
  if (world.rank() != 0) {
    auto* input_matrix = reinterpret_cast<std::vector<std::vector<double>>*>(taskData->inputs[0]);
    if (input_matrix == nullptr || input_matrix->empty()) {
      local_valid = 0;  // Матрица пустая или некорректная
    }
  }

  // Шаг 2: Синхронизация всех результатов
  size_t global_valid = 0;
  boost::mpi::reduce(world, local_valid, global_valid, std::plus<>(), 0);

  // Шаг 3: Процесс с рангом 0 передает результат всем остальным процессам
  boost::mpi::broadcast(world, global_valid, 0);

  // Если хотя бы один процесс не прошел валидацию, возвращаем false
  return global_valid > 0;
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
