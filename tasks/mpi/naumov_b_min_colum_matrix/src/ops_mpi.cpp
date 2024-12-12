#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <limits>
#include <vector>

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    for (unsigned j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[i * taskData->inputs_count[1] + j];
    }
  }

  res.resize(taskData->inputs_count[1], std::numeric_limits<int>::max());
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
      res[j] = std::min(res[j], input_[i][j]);
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[1] * taskData->inputs_count[0], 0);
    auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      for (unsigned j = 0; j < taskData->inputs_count[1]; j++) {
        input_[i + j * taskData->inputs_count[0]] = temp[j + i * taskData->inputs_count[1]];
      }
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) {
      return false;
    }
    if (taskData->inputs_count.size() < 2) {
      return false;
    }
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      return false;
    }
    if (taskData->outputs_count[0] != taskData->inputs_count[1]) {
      return false;
    }
    return true;
  }
  return true;
}
bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rows = 0;
  int cols = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, cols, 0);

  int delta = cols / world.size();
  int extra = cols % world.size();

  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, extra, 0);

  int num_columns = delta + (world.rank() < extra ? 1 : 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_start_col = proc * delta + std::min(proc, extra);
      int proc_num_columns = delta + (proc < extra ? 1 : 0);
      world.send(proc, 0, input_.data() + proc_start_col * rows, proc_num_columns * rows);
    }

    local_vector_ = std::vector<int>(input_.begin(), input_.begin() + num_columns * rows);

  } else {
    local_vector_ = std::vector<int>(num_columns * rows);
    world.recv(0, 0, local_vector_.data(), num_columns * rows);
  }

  std::vector<int> local_res(num_columns, std::numeric_limits<int>::max());

  for (int i = 0; i < num_columns; ++i) {
    for (int j = 0; j < rows; ++j) {
      local_res[i] = std::min(local_res[i], local_vector_[j + i * rows]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> temp(delta, 0);
    res.insert(res.end(), local_res.begin(), local_res.end());
    for (int i = 1; i < world.size(); i++) {
      int recv_num_columns = delta + (i < extra ? 1 : 0);
      temp.resize(recv_num_columns);
      world.recv(i, 0, temp.data(), recv_num_columns);
      res.insert(res.end(), temp.begin(), temp.end());
    }
  } else {
    world.send(0, 0, local_res.data(), num_columns);
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}
