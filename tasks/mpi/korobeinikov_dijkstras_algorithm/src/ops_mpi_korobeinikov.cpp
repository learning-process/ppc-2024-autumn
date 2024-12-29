// Copyright 2024 Korobeinikov Arseny
#include "mpi/korobeinikov_dijkstras_algorithm/include/ops_mpi_korobeinikov.hpp"

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  values.reserve(taskData->inputs_count[0]);
  auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], values.begin());

  col.reserve(taskData->inputs_count[1]);
  auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[1]);
  std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[1], col.begin());

  RowIndex.reserve(taskData->inputs_count[2]);
  auto *tmp_ptr_3 = reinterpret_cast<int *>(taskData->inputs[2]);
  std::copy(tmp_ptr_3, tmp_ptr_3 + taskData->inputs_count[2], RowIndex.begin());

  size = *reinterpret_cast<int *>(taskData->inputs[3]);
  sv = *reinterpret_cast<int *>(taskData->inputs[4]);

  res = std::vector<int>(size, 0);
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output

  bool flag = true;
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    if (taskData->inputs[2][i] < 0) {
      flag = false;
    }
  }
  return taskData->inputs.size() == 5 && taskData->inputs_count.size() == 5 && taskData->outputs.size() == 1 &&
         taskData->outputs_count.size() == 1 && flag && *(reinterpret_cast<int *>(taskData->inputs[4])) >= 0 &&
         (*reinterpret_cast<int *>(taskData->inputs[4])) < (*reinterpret_cast<int *>(taskData->inputs[3])) &&
         (int)taskData->outputs_count[0] == *reinterpret_cast<int *>(taskData->inputs[3]) &&
         (int)taskData->inputs_count[2] == *reinterpret_cast<int *>(taskData->inputs[3]) + 1;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential::run() {
  internal_order_test();

  std::vector<bool> visited(size, false);
  std::vector<int> D(size, std::numeric_limits<int>::max());
  D[sv] = 0;
  for (int i = 0; i < size; i++) {
    int min = std::numeric_limits<int>::max();
    int index = -1;
    for (int j = 0; j < size; j++) {
      if (!visited[j] && D[j] < min) {
        min = D[j];
        index = j;
      }
    }
    if (index == -1) break;
    visited[index] = true;
    for (int j = RowIndex[index]; j < RowIndex[index + 1]; j++) {
      int v = col[j];
      int weight = values[j];

      if (!visited[v] && (D[index] + weight < D[v])) {
        D[v] = D[index] + weight;
      }
    }
  }
  res = D;
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init value for input and output
    values.resize(taskData->inputs_count[0]);
    auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
    std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], values.begin());

    col.resize(taskData->inputs_count[1]);
    auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[1]);
    std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[1], col.begin());

    RowIndex.resize(taskData->inputs_count[2]);
    auto *tmp_ptr_3 = reinterpret_cast<int *>(taskData->inputs[2]);
    std::copy(tmp_ptr_3, tmp_ptr_3 + taskData->inputs_count[2], RowIndex.begin());

    size = *reinterpret_cast<int *>(taskData->inputs[3]);
    sv = *reinterpret_cast<int *>(taskData->inputs[4]);
  }
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    bool flag = true;
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      if (taskData->inputs[2][i] < 0) {
        flag = false;
      }
    }

    return taskData->inputs.size() == 5 && taskData->inputs_count.size() == 5 && taskData->outputs.size() == 1 &&
           taskData->outputs_count.size() == 1 && flag && *(reinterpret_cast<int *>(taskData->inputs[4])) >= 0 &&
           (*reinterpret_cast<int *>(taskData->inputs[4])) < (*reinterpret_cast<int *>(taskData->inputs[3])) &&
           (int)taskData->outputs_count[0] == *reinterpret_cast<int *>(taskData->inputs[3]) &&
           (int)taskData->inputs_count[2] == *reinterpret_cast<int *>(taskData->inputs[3]) + 1;
  }
  return true;
}

struct ComparePairs {
  std::pair<int, int> operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const {
    if (a.first < b.first) return a;
    if (a.first > b.first) return b;
    return a;
  }
};

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, size, 0);
  broadcast(world, sv, 0);
  int count_edges = values.size();
  broadcast(world, count_edges, 0);

  if (world.rank() != 0) {
    values.resize(count_edges);
    col.resize(count_edges);
    RowIndex.resize(size + 1);
  }

  boost::mpi::broadcast(world, values.data(), values.size(), 0);
  boost::mpi::broadcast(world, col.data(), col.size(), 0);
  boost::mpi::broadcast(world, RowIndex.data(), RowIndex.size(), 0);

  int num_use_proc = std::min(world.size(), size);
  if (num_use_proc == 0) {
    return true;
  }
  int delta = size % num_use_proc == 0 ? (size / num_use_proc) : ((size / num_use_proc) + 1);

  int begin_for_proc = world.rank() * delta;
  int end_for_proc = std::min(size, delta * (world.rank() + 1));

  std::vector<bool> visited(size, false);
  res = std::vector<int>(size, std::numeric_limits<int>::max());
  res[sv] = 0;

  for (int k = 0; k < size; k++) {
    int local_min = std::numeric_limits<int>::max();
    int local_index = -1;

    for (int i = begin_for_proc; i < end_for_proc; i++) {
      if (!visited[i] && res[i] < local_min) {
        local_min = res[i];
        local_index = i;
      }
    }
    std::pair<int, int> local_pair = {local_min, local_index};
    std::pair<int, int> global_pair = {std::numeric_limits<int>::max(), -1};

    boost::mpi::all_reduce(world, local_pair, global_pair, ComparePairs());

    if (global_pair.first == std::numeric_limits<int>::max()) {
      break;
    }

    visited[global_pair.second] = true;

    for (int j = RowIndex[global_pair.second]; j < RowIndex[global_pair.second + 1]; j++) {
      int v = col[j];
      int w = values[j];

      if (!visited[v] && (res[global_pair.second] + w < res[v])) {
        res[v] = res[global_pair.second] + w;
      }
    }
  }

  return true;
}

bool korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  }
  return true;
}
