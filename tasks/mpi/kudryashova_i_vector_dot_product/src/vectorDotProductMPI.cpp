#include "mpi/kudryashova_i_vector_dot_product/include/vectorDotProductMPI.hpp"

#include <boost/mpi.hpp>
#include <random>
#include <thread>

static int seedOffset = 0;
using namespace std::chrono_literals;

std::vector<int> kudryashova_i_vector_dot_product_mpi::getRandomVector(unsigned long size) {
  std::vector<int> vector(size);
  std::srand(static_cast<unsigned>(time(NULL)) + ++seedOffset);
  for (unsigned long i = 0; i < size; ++i) {
    vector[i] = std::rand() % 10 + 1;
  }
  return vector;
}

int kudryashova_i_vector_dot_product_mpi::vectorDotProduct(const std::vector<int>& vector1,
                                                           const std::vector<int>& vector2) {
  long long result = 0;
  for (unsigned long i = 0; i < vector1.size(); i++) result += vector1[i] * vector2[i];
  return result;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs.size());
  for (unsigned long i = 0; i < input_.size(); ++i) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[i]);
    input_[i] = std::vector<int>(taskData->inputs_count[i]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[i], input_[i].begin());
  }
  result = 0;
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
         taskData->outputs.size() == 1;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_[0].size(); i++) {
    result += input_[1][i] * input_[0][i];
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_.resize(taskData->inputs.size());
    for (size_t i = 0; i < taskData->inputs.size(); ++i) {
      input_[i].resize(taskData->inputs_count[i]);

      int* source_ptr = reinterpret_cast<int*>(taskData->inputs[i]);

      if (source_ptr != nullptr) {
        std::copy(source_ptr, source_ptr + taskData->inputs_count[i], input_[i].begin());
      } else
        return false;
    }
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, input_[0].data() + proc * delta, delta);
      world.send(proc, 1, input_[1].data() + proc * delta, delta);
    }
  }
  local_input1_.resize(delta);
  local_input2_.resize(delta);

  if (world.rank() == 0) {
    std::copy(input_[0].begin(), input_[0].begin() + delta, local_input1_.begin());
    std::copy(input_[1].begin(), input_[1].begin() + delta, local_input2_.begin());
  } else {
    world.recv(0, 0, local_input1_.data(), delta);
    world.recv(0, 1, local_input2_.data(), delta);
  }
  result = 0;
  return true;
}

//bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::pre_processing() {
//  internal_order_test();
//
//  // Вычисляем количество данных для каждого процесса
//  std::vector<int> counts(world.size());
//  for (int i = 0; i < world.size(); ++i) {
//    counts[i] = taskData->inputs_count[i] / world.size();
//  }
//
//  // Вычисляем смещения для ручного распределения данных
//  std::vector<int> displacements(world.size());
//  displacements[0] = 0;
//  for (int i = 1; i < world.size(); ++i) {
//    displacements[i] = displacements[i - 1] + counts[i - 1];
//  }
//
//  // Проверяем, что количество данных достаточно для распределения
//  //if (world.rank() == 0) {
//  //  if (taskData->inputs_count[0] < world.size()) {
//  //    return false;
//  //  }
//  //}
//
//  // Заполняем input_ на всех процессах
//  input_.resize(taskData->inputs.size());
//  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
//    input_[i].resize(taskData->inputs_count[i]);
//    std::copy(taskData->inputs[i], taskData->inputs[i] + taskData->inputs_count[i], input_[i].begin());
//  }
//
//  // Инициализация локальных буферов
//  local_input1_.resize(counts[world.rank()]);
//  local_input2_.resize(counts[world.rank()]);
//
//  // Ручное распределение данных
//  if (world.rank() == 0) {
//    // Процесс 0 распределяет данные
//    for (int proc = 0; proc < world.size(); ++proc) {
//      if (proc != 0) {
//        world.send(proc, 0, input_[0].data() + displacements[proc], counts[proc]);
//        world.send(proc, 1, input_[1].data() + displacements[proc], counts[proc]);
//      } else {
//        // Процесс 0 копирует свою часть данных
//        std::copy(input_[0].begin() + displacements[proc], input_[0].begin() + displacements[proc] + counts[proc],
//                  local_input1_.begin());
//        std::copy(input_[1].begin() + displacements[proc], input_[1].begin() + displacements[proc] + counts[proc],
//                  local_input2_.begin());
//      }
//    }
//  } else {
//    // Остальные процессы получают данные
//    world.recv(0, 0, local_input1_.data(), counts[world.rank()]);
//    world.recv(0, 1, local_input2_.data(), counts[world.rank()]);
//  }
//
//  result = 0;
//  return true;
//}

//
//
//bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::validation() {
//  internal_order_test();
//  if (world.rank() == 0) {
//    return (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
//           (taskData->inputs_count[0] == taskData->inputs_count[1]) && taskData->outputs_count[0] == 1 &&
//           (taskData->outputs.size() == taskData->outputs_count.size()) && taskData->outputs.size() == 1;
//  }
//  return true;
//}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs.size() != taskData->inputs_count.size()) || (taskData->inputs.size() != 2) ||
        (taskData->inputs_count[0] != taskData->inputs_count[1]) || (taskData->outputs_count[0] != 1) ||
        (taskData->outputs.size() != taskData->outputs_count.size()) || (taskData->outputs.size() != 1)) {
      return false;
    }
  }
  return true;
}

bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int local_result = std::inner_product(local_input1_.begin(), local_input1_.end(), local_input2_.begin(), 0);

  std::vector<int> full_results;
  gather(world, local_result, full_results, 0);

  if (world.rank() == 0) {
    result = std::accumulate(full_results.begin(), full_results.end(), 0);
  }
  return true;
}

 bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
   internal_order_test();
   if (world.rank() == 0) {
     reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
   }
   return true;
 }
//bool kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
//  internal_order_test();
//  if (world.rank() == 0) {
//    if (taskData->outputs.size() > 0) {
//      reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
//    } else {
//      return false;
//    }
//  }
//  return true;
//}