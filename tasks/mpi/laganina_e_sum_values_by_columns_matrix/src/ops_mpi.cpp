#include "mpi/laganina_e_sum_values_by_columns_matrix/include/ops_mpi.hpp"

#include <thread>
#include <vector>

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  m = taskData->inputs_count[1];
  n = taskData->inputs_count[2];
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = ptr[i];
  }
  res_ = std::vector<int>(n, 0);
  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[2] != taskData->outputs_count[0]) {
    return false;
  };
  if (taskData->inputs_count[1] < 1 || taskData->inputs_count[2] < 1) {
    return false;
  }
  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[2]) {
    return false;
  }
  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int j = 0; j < n; j++) {
    int sum = 0;
    for (int i = 0; i < m; i++) {
      sum += input_[i * n + j];
    }
    res_[j] = sum;
  }
  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < n; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int size = 0;
  unsigned int delta = 0;

  if (world.rank() == 0) {
    m = taskData->inputs_count[1];
    n = taskData->inputs_count[2];
    size = n * m;
    if (size % world.size() == 0) {
      delta = size / world.size();
    } else {
      delta = size / world.size() + 1;
    }
    input_ = std::vector<int>(delta * world.size());
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < n; i++) {
      for (int k = i * m, r = i; r < size; r += n, k++) {
        input_[k] = tmp_ptr[r];
      }
    }
  }

  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[2] != taskData->outputs_count[0]) {
      return false;
    };
    if (taskData->inputs_count[1] < 1 || taskData->inputs_count[2] < 1) {
      return false;
    }
    if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[2]) {
      return false;
    }
    return true;
  }
  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int size = 0;
  unsigned int delta = 0;

  if (world.rank() == 0) {
    n = taskData->inputs_count[1];
    m = taskData->inputs_count[2];
    size = n * m;
    if (size % world.size() == 0) {
      delta = size / world.size();
    } else {
      delta = size / world.size() + 1;
    }
  }

  broadcast(world, m, 0);
  broadcast(world, n, 0);
  broadcast(world, delta, 0);

  local_input_ = std::vector<int>(delta);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);
  res_.resize(m);
  unsigned int last = 0;

  if (world.rank() == world.size() - 1) {
    last = local_input_.size() * world.size() - n * m;
  }
  unsigned int id = world.rank() * local_input_.size() / n;

  for (unsigned int i = 0; i < id; i++) {
    reduce(world, 0, res_[i], std::plus(), 0);
  }

  delta = std::min(local_input_.size(), n - world.rank() * local_input_.size() % n);
  int l_res = std::accumulate(local_input_.begin(), local_input_.begin() + delta, 0);
  reduce(world, l_res, res_[id], std::plus(), 0);
  id++;
  unsigned int k = 0;

  while (local_input_.begin() + delta + k * n < local_input_.end() - last) {
    l_res = std::accumulate(local_input_.begin() + delta + k * n,
                            std::min(local_input_.end(), local_input_.begin() + delta + (k + 1) * n), 0);
    reduce(world, l_res, res_[id], std::plus(), 0);
    k++;
    id++;
  }

  for (unsigned int i = id; i < res_.size(); i++) {
    reduce(world, 0, res_[i], std::plus(), 0);
  }

  return true;
}

bool laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < m; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
  }
  return true;
}
