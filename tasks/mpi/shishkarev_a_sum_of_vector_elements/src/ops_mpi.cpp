// Copyright 2023 Nesterov Alexander
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

std::vector<int> shishkarev_a_sum_of_vector_elements_mpi::getRandomVector(int vector_size) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0, 99);
  std::vector<int> random_vector(vector_size);
  std::generate(random_vector.begin(), random_vector.end(), [&]() { return distribution(generator); });
  return random_vector;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::pre_processing() {
  internal_order_test();

  // Скопировать входные данные в локальный вектор
  input_vector = std::vector<int>(taskData->inputs_count[0]);
  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_vector[i] = input_ptr[i];
  }

  // Инициализация результата
  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::validation() {
  internal_order_test();

  // Убедимся, что выходной буфер имеет правильный размер
  return taskData->outputs_count[0] == 1;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::run() {
  internal_order_test();

  // Подсчитываем сумму элементов вектора
  result = std::accumulate(input_vector.cbegin(), input_vector.cend(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumSequential::post_processing() {
  internal_order_test();

  // Передаем результат в выходной буфер
  *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::pre_processing() {
  internal_order_test();

  // Определяем размер данных для разделения
  int world_id = world.rank();
  int world_size = world.size();
  unsigned int n = 0;

  if (world_id == 0) {
    // Инициализируем вводные данные
    n = taskData->inputs_count[0];
    input_vector = std::vector<int>(n);
    int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    memcpy(input_vector.data(), input_ptr, sizeof(int) * n);
  }

  // распространяем размеры данных
  boost::mpi::broadcast(world, n, 0);

  // Определяем размер подзадач
  unsigned int vector_send_size = n / world_size;
  unsigned int local_size = n % world_size;
  std::vector<int> send_counts(world_size, vector_send_size);
  std::vector<int> disp(world_size, 0);

  for (unsigned int i = 0; i < static_cast<unsigned int>(world_size); ++i) {
    if (i < local_size) {
      ++send_counts[i];
    }
    if (i > 0) {
      disp[i] = disp[i - 1] + send_counts[i - 1];
    }
  }

  auto local_vector_size = static_cast<unsigned int>(send_counts[world_id]);
  local_vector.resize(local_vector_size);

  // Разделяем данные на процессы
  boost::mpi::scatterv(world, input_vector.data(), send_counts, disp, local_vector.data(), local_vector_size, 0);

  local_sum = 0;
  result = 0;
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::run() {
  internal_order_test();

  // Считаем локальную сумму
  local_sum = std::accumulate(local_vector.begin(), local_vector.end(), 0);

  // Суммируем результаты всех процессов
  boost::mpi::reduce(world, local_sum, result, std::plus<>(), 0);
  return true;
}

bool shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  }

  return true;
}
