#include "mpi/plekhanov_d_allreduce_mine/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::vector<int> plekhanov_d_allreduce_mine_mpi::getRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    int value = 0;
    do {
      value = gen() % 1000 + 100;
    } while (value <= 0);
    vec[i] = value;
  }
  return vec;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors

  columnCount = taskData->inputs_count[1];
  rowCount = taskData->inputs_count[2];

  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  inputData_.assign(tempPtr, tempPtr + taskData->inputs_count[0]);

  resultData_ = std::vector<int>(columnCount, 0);
  countAboveMin_ = std::vector<int>(columnCount, 0);

  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
          taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  // Находим минимум в каждом столбце
  for (int column = 0; column < columnCount; column++) {
    int columnMin = inputData_[column];  // Инициализируем минимум первым элементом столбца
    for (int row = 1; row < rowCount; row++) {
      if (inputData_[row * columnCount + column] < columnMin) {
        columnMin = inputData_[row * columnCount + column];
      }
    }
    resultData_[column] = columnMin;  // Сохраняем минимум для столбца
  }

  // Считаем количество элементов, которые больше минимума
  for (int column = 0; column < columnCount; column++) {
    for (int row = 0; row < rowCount; row++) {
      if (inputData_[row * columnCount + column] > resultData_[column]) {
        countAboveMin_[column]++;
      }
    }
  }
  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < columnCount; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = countAboveMin_[i];
  }
  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskMyOwnParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    columnCount = taskData->inputs_count[1];
    rowCount = taskData->inputs_count[2];
  }

  if (world.rank() == 0) {
    // init vectors
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    inputData_.assign(tempPtr, tempPtr + taskData->inputs_count[0]);
  } else {
    inputData_ = std::vector<int>(columnCount * rowCount, 0);
  }

  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskMyOwnParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[1] != 0 && taskData->inputs_count[2] != 0 && !taskData->inputs.empty() &&
            taskData->inputs_count[0] > 0 && (taskData->inputs_count[1] == taskData->outputs_count[0]));
  }
  return true;
}

template <typename T>
void plekhanov_d_allreduce_mine_mpi::TestMPITaskMyOwnParallel::my_all_reduce(const boost::mpi::communicator& world,
                                                                          const T* in_values, T* out_values, int n) {
  int root = world.rank();
  std::vector<T> left_values(n), right_values(n);

  int left_child = 2 * root + 1;
  int right_child = 2 * root + 2;

  // Инициализация out_values значениями из in_values
  std::copy(in_values, in_values + n, out_values);

  // Получение значений от левого и правого потомков
  if (left_child < world.size()) {
    world.recv(left_child, 0, left_values.data(), n);
  }

  if (right_child < world.size()) {
    world.recv(right_child, 0, right_values.data(), n);
  }

  // Выполнение редукции (поиск минимума)
  for (int i = 0; i < n; ++i) {
    if (left_child < world.size()) {
      out_values[i] = std::min(out_values[i], left_values[i]);
    }
    if (right_child < world.size()) {
      out_values[i] = std::min(out_values[i], right_values[i]);
    }
  }

  // Отправка уменьшенных значений родителю, если нужно
  if (root != 0) {
    int parent = (root - 1) / 2;
    world.send(parent, 0, out_values, n);
    world.recv(parent, 0, out_values, n);  // Получение финальных уменьшенных значений
  }

  // Распространение уменьшенных значений на потомков (если есть)
  if (left_child < world.size()) {
    world.send(left_child, 0, out_values, n);
  }

  if (right_child < world.size()) {
    world.send(right_child, 0, out_values, n);
  }
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskMyOwnParallel::run() {
  internal_order_test();

  broadcast(world, columnCount, 0);
  broadcast(world, rowCount, 0);

  if (world.rank() != 0) {
    inputData_ = std::vector<int>(columnCount * rowCount, 0);
  }
  broadcast(world, inputData_.data(), columnCount * rowCount, 0);

  int delta = columnCount / world.size();
  int extra = columnCount % world.size();
  int startColumn = delta * world.rank();
  int lastColumn = (world.rank() == world.size() - 1) ? (startColumn + delta + extra) : (startColumn + delta);

  std::vector<int> localMin(columnCount, INT_MAX);
  for (int column = startColumn; column < lastColumn; column++) {
    for (int row = 0; row < rowCount; row++) {
      int coordinate = row * columnCount + column;
      localMin[column] = std::min(localMin[column], inputData_[coordinate]);
    }
  }
  
  // Собираем минимумы с помощью all_reduce
 
  resultData_.resize(columnCount);
  my_all_reduce(world, localMin.data(), resultData_.data(), columnCount);
 
  // Считаем количество значений, которые больше минимума
  std::vector<int> localCount(columnCount, 0);
  for (int column = startColumn; column < lastColumn; column++) {
    for (int row = 0; row < rowCount; row++) {
      int coordinate = row * columnCount + column;
      if (inputData_[coordinate] > resultData_[column]) {
        localCount[column]++;
        
      }
    }
    /*std::cout << localCount[column] << "-----------" << resultData_[column] << std::endl;*/
  }
  // Суммируем локальные счетчики
  countAboveMin_.resize(columnCount, 0);
  boost::mpi::reduce(world, localCount.data(), columnCount, countAboveMin_.data(), std::plus<>(), 0);
  
  return true;
}

bool plekhanov_d_allreduce_mine_mpi::TestMPITaskMyOwnParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < columnCount; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = countAboveMin_[i];
    }
  }
  return true;
}


// cd C:\Users\Danie\Desktop\ppc-2024-autumn\build\bin
// mpiexec -n 4 mpi_func_tests.exe