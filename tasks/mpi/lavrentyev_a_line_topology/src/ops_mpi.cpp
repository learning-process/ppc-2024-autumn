#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <vector>

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int start_proc = static_cast<int>(taskData->inputs_count[0]);
  int num_of_elems = static_cast<int>(taskData->inputs_count[2]);

  if (world.rank() == start_proc) {
    const auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    if (input_data != nullptr) {
      data.assign(input_data, input_data + num_of_elems);
    }
    path.clear();
    path.push_back(world.rank());
  }
  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) {
    return false;
  }

  int start_proc = static_cast<int>(taskData->inputs_count[0]);
  int end_proc = static_cast<int>(taskData->inputs_count[1]);
  int num_elems = static_cast<int>(taskData->inputs_count[2]);

  if (start_proc < 0 || start_proc >= world.size() || end_proc < 0 || end_proc >= world.size() || num_elems <= 0) {
    return false;
  }

  if (world.rank() == start_proc) {
    if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      return false;
    }
  }

  if (world.rank() == end_proc) {
    if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs[1] == nullptr) {
      return false;
    }
  }

  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  // Приводим start_proc и end_proc к типу int, так как world.rank() возвращает int
  int start_proc = static_cast<int>(taskData->inputs_count[0]);
  int end_proc = static_cast<int>(taskData->inputs_count[1]);

  // Если start_proc и end_proc совпадают, возвращаем true
  if (start_proc == end_proc) {
    return true;
  }

  int rank = world.rank();  // Получаем rank как int для удобства
  // Проверяем, находится ли rank внутри допустимого диапазона
  if (rank < start_proc || rank > end_proc) {
    return true;
  }

  std::vector<boost::mpi::request> requests;

  if (rank == start_proc) {
    // Отправляем данные и путь, если rank == start_proc
    requests.push_back(world.isend(rank + 1, 0, reinterpret_cast<uint8_t*>(data.data()),
                                   data.size() * sizeof(int)));  // Отправляем данные как uint8_t* (байтовый массив)

    // Преобразуем path в uint8_t* и передаем
    requests.push_back(world.isend(rank + 1, 1, reinterpret_cast<uint8_t*>(path.data()),
                                   path.size() * sizeof(int)));  // Отправляем path как uint8_t* (байтовый массив)
  } else {
    // Принимаем данные и путь, если rank > start_proc
    boost::mpi::request recv_data_req = world.irecv(rank - 1, 0, reinterpret_cast<uint8_t*>(data.data()),
                                                    data.size() * sizeof(int));  // Принимаем данные как uint8_t*
    boost::mpi::request recv_path_req = world.irecv(rank - 1, 1, reinterpret_cast<uint8_t*>(path.data()),
                                                    path.size() * sizeof(int));  // Принимаем путь как uint8_t*

    // Ожидаем завершения получения данных
    recv_data_req.wait();
    recv_path_req.wait();

    // Добавляем текущий rank в путь
    path.push_back(rank);

    // Отправляем данные и путь следующему процессу, если rank меньше end_proc
    if (rank < end_proc) {
      requests.push_back(world.isend(rank + 1, 0, reinterpret_cast<uint8_t*>(data.data()),
                                     data.size() * sizeof(int)));  // Отправляем данные как uint8_t*
      requests.push_back(world.isend(rank + 1, 1, reinterpret_cast<uint8_t*>(path.data()),
                                     path.size() * sizeof(int)));  // Отправляем путь как uint8_t*
    }
  }

  // Ожидаем завершения всех запросов
  boost::mpi::wait_all(requests.begin(), requests.end());

  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int end_proc = static_cast<int>(taskData->inputs_count[1]);

  if (world.rank() == end_proc) {
    auto* data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    if (data_ptr != nullptr) {
      std::copy(data.begin(), data.end(), data_ptr);
    }
    auto* path_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
    if (path_ptr != nullptr) {
      std::copy(path.begin(), path.end(), path_ptr);
    }
  }

  return true;
}
