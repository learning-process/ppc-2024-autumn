// Copyright 2023 Nesterov Alexander
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> naumov_b_min_colum_matrix_mpi::getRandomVector(int size) {
  std::vector<int> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = rand() % 201 - 100;  // Генерируем числа от -100 до 100
  }
  return vec;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  // Init matrix
  input_.resize(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    for (unsigned j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[i * taskData->inputs_count[1] + j];
    }
  }

  // Init value for output
  res.resize(taskData->inputs_count[1]);
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  size_t numRows = input_.size();
  size_t numCols = input_[0].size();

  for (size_t j = 0; j < numCols; j++) {
    res[j] = input_[0][j];
    for (size_t i = 1; i < numRows; i++) {
      if (input_[i][j] < res[j]) {
        res[j] = input_[i][j];
      }
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rows = 0;
  int cols = 0;

  // Получаем количество строк и столбцов
  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  // Распространяем количество строк и столбцов среди всех процессов
  broadcast(world, rows, 0);
  broadcast(world, cols, 0);

  int delta = rows / world.size();
  int extra = rows % world.size();

  // Определяем количество строк, которые будет обрабатывать текущий процесс
  int local_rows = (world.rank() < extra) ? (delta + 1) : delta;
  local_input_.resize(local_rows, std::vector<int>(cols));  // Инициализация вектора

  if (world.rank() == 0) {
    // Инициализация матрицы
    input_.resize(rows, std::vector<int>(cols));
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        input_[i][j] = input_matrix[i * cols + j];
      }
    }

    // Отправка данных другим процессам
    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * delta + std::min(proc, extra);
      int num_rows = delta + (proc < extra ? 1 : 0);
      for (int r = start_row; r < start_row + num_rows; r++) {
        world.send(proc, 0, input_[r].data(), cols);
      }
    }
  }

  // Получаем данные для текущего процесса
  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_rows, local_input_.begin());
  } else {
    for (int r = 0; r < local_rows; r++) {
      world.recv(0, 0, local_input_[r].data(), cols);
    }
  }

  // Инициализация вектора для хранения результатов
  res.resize(cols);
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (!taskData->inputs.empty() && !taskData->outputs.empty()) && (taskData->inputs_count.size() >= 2) &&
           (taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
           (taskData->outputs_count[0] == taskData->inputs_count[1]);
  }

  return true;  // Остальные процессы просто возвращают true
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int local_rows = local_input_.size();
  if (local_rows == 0) return true;  // Если нет локальных строк, завершаем

  int numCols = local_input_[0].size();
  std::vector<int> local_minima(numCols, std::numeric_limits<int>::max());

  // Поиск локальных минимумов по столбцам
  for (int j = 0; j < numCols; ++j) {
    for (int i = 0; i < local_rows; ++i) {
      int value = local_input_.at(i).at(j);
      if (value < local_minima[j]) {
        local_minima[j] = value;
      }
    }
  }

  // Сбор всех локальных минимумов на процессе с rank 0
  std::vector<int> all_minima(world.size() * numCols);
  MPI_Gather(local_minima.data(), numCols, MPI_INT, all_minima.data(), numCols, MPI_INT, 0, MPI_COMM_WORLD);

  if (world.rank() == 0) {
    res.resize(numCols);
    std::fill(res.begin(), res.end(), std::numeric_limits<int>::max());

    for (int i = 0; i < world.size(); ++i) {
      for (int j = 0; j < numCols; ++j) {
        int value = all_minima[i * numCols + j];
        if (value < res[j]) {
          res[j] = value;
        }
      }
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res.begin(), res.end(), output_matrix);  // Записываем глобальные минимумы
  }

  return true;
}
