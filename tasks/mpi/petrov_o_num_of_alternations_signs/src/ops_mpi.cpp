#include "mpi/petrov_o_num_of_alternations_signs/include/ops_mpi.hpp"

#include <boost/mpi/datatype.hpp>
#include <vector>

using namespace std::chrono_literals;

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::pre_processing() {
  internal_order_test();
  this->res = 0;
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::validation() {
  internal_order_test();

  int input_size = 0;

  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
  }

  boost::mpi::broadcast(world, input_size,
                        0);  // Без broadcast не получается сделать проверку на избыточное число процессов

  if (input_size < world.size()) {
    return false;
  }

  if (world.rank() != 0) return true;
  return taskData->outputs_count[0] == 1;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::run() {
  internal_order_test();

  int input_size = 0;

  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
  }

  boost::mpi::broadcast(world, input_size, 0);  // Чтобы значение было доступно везде

  if (world.rank() == 0) {
    const int* input = reinterpret_cast<int*>(taskData->inputs[0]);
    this->input_.resize(input_size);
    std::copy(input, input + input_size, std::begin(this->input_));

    std::vector<int> distribution(world.size());
    std::vector<int> displacement(world.size());

    int chunk_size = input_size / world.size();
    int remainder = input_size % world.size();

    for (int i = 0; i < world.size(); ++i) {
      distribution[i] =
          chunk_size +
          static_cast<int>(i < remainder);  // Добавим 1 элемент к некоторым чанкам, чтобы распределить остаток
      displacement[i] =
          (i == 0) ? 0 : displacement[i - 1] + distribution[i - 1];  // Смещение текущего ненулевого блока равно
                                                                     // смещению предыдущего блока + его размеру
    }

    chunk.resize(distribution[world.rank()]);

    boost::mpi::scatterv(world, input, distribution, displacement, chunk.data(), distribution[world.rank()], 0);

  } else {
    int chunk_size = input_size / world.size();
    int remainder = input_size % world.size();

    int distribution = chunk_size + static_cast<int>(world.rank() < remainder);

    chunk.resize(distribution);  // Зарезервируем необходимое место под данные

    int input;  // Функция при тестировании clang-tidy требует наличия указателя на input, который в дальнейшем не
                // используется. Поэтому создаем фиктивную переменную
    boost::mpi::scatterv(world, &input, chunk.data(), distribution, 0);
  }

  auto local_res = 0;

  for (size_t i = 1; i < chunk.size(); i++) {
    if ((chunk[i] < 0) ^ (chunk[i - 1] < 0)) {
      local_res++;
    }
  }

  /*Проверка чередования граничных элементов*/
  int last_element = chunk.back();
  int next_element = 0;

  if (world.rank() < world.size() - 1) {  // Если это не последний процесс
    world.send(world.rank() + 1, 0, last_element);
  }

  if (world.rank() > 0) {  // Если это не первый процесс
    world.recv(world.rank() - 1, 0, next_element);
    if ((chunk.front() < 0) ^ (next_element < 0)) {
      local_res++;
    }
  }

  boost::mpi::reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  }
  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::pre_processing() {
  internal_order_test();

  const auto input_size = taskData->inputs_count[0];

  const int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  this->input_.resize(input_size);
  std::copy(input, input + input_size, std::begin(this->input_));

  this->res = 0;  // Обнуляем счетчик каждый новый запуск

  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;  // Проверяем, что на выходе ожидается одно число
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::run() {
  internal_order_test();

  if (input_.size() > 1) {
    for (size_t i = 1; i < input_.size(); i++) {
      if ((input_[i] < 0) ^ (input_[i - 1] < 0)) {
        this->res++;
      }
    }
  }

  return true;
}

bool petrov_o_num_of_alternations_signs_mpi::SequentialTask::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;  // Передаем резульбтат
  return true;
}
