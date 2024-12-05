// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_custom_reduce/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool ermilova_d_custom_reduce_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];

  input_.resize(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; i++) {
    auto* tpr_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) {
      input_[i][j] = tpr_ptr[j];
    }
  }
  // Init value for output
  res = INT_MAX;
  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < input_.size(); i++) {
    for (size_t j = 0; j < input_[i].size(); j++) {
      if (res > input_[i][j]) {
        res = input_[i][j];
      }
    }
  }
  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * cols);

    for (int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_ptr[j];
      }
    }
  }
  res = INT_MAX;
  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !(taskData->inputs.empty());
  }
  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int extra = 0;

  if (world.rank() == 0) {
    delta = rows * cols / world.size();
    extra = rows * cols % world.size();
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc + extra, delta);
    }
  }

  local_input_ = std::vector<int>(delta);

  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + extra);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  int local_min = INT_MAX;
  if (!local_input_.empty()) {
    local_min = *std::min_element(local_input_.begin(), local_input_.end());
  }

  ermilova_d_custom_reduce_mpi::CustomReduce(&local_min, &res, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  return true;
}

bool ermilova_d_custom_reduce_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void ermilova_d_custom_reduce_mpi::CustomReduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int type_size;
  MPI_Type_size(datatype, &type_size);

  if (rank == root) {
    std::memcpy(recvbuf, sendbuf, count * type_size);
  }

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      if (rank + step < size) {
        std::vector<char> buffer(count * type_size);
        MPI_Recv(buffer.data(), count, datatype, rank + step, 0, comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++) {
          if (datatype == MPI_INT) {
            int* recv = reinterpret_cast<int*>(recvbuf);
            int* buf = reinterpret_cast<int*>(buffer.data());
            if (op == MPI_SUM) {
              recv[i] += buf[i];
            } else if (op == MPI_MIN) {
              recv[i] = std::min(recv[i], buf[i]);
            } else if (op == MPI_MAX) {
              recv[i] = std::max(recv[i], buf[i]);
            } else {
              throw std::runtime_error("Unsupported MPI_Op in CustomReduce");
            }
          } else if (datatype == MPI_FLOAT) {
            float* recv = reinterpret_cast<float*>(recvbuf);
            float* buf = reinterpret_cast<float*>(buffer.data());
            if (op == MPI_SUM) {
              recv[i] += buf[i];
            } else if (op == MPI_MIN) {
              recv[i] = std::min(recv[i], buf[i]);
            } else if (op == MPI_MAX) {
              recv[i] = std::max(recv[i], buf[i]);
            } else {
              throw std::runtime_error("Unsupported MPI_Op in CustomReduce");
            }
          } else if (datatype == MPI_DOUBLE) {
            double* recv = reinterpret_cast<double*>(recvbuf);
            double* buf = reinterpret_cast<double*>(buffer.data());
            if (op == MPI_SUM) {
              recv[i] += buf[i];
            } else if (op == MPI_MIN) {
              recv[i] = std::min(recv[i], buf[i]);
            } else if (op == MPI_MAX) {
              recv[i] = std::max(recv[i], buf[i]);
            } else {
              throw std::runtime_error("Unsupported MPI_Op in CustomReduce");
            }
          }
        }
      }
    } else if (rank % step == 0) {
      MPI_Send(recvbuf, count, datatype, rank - step, 0, comm);
    }
    step *= 2;
  }
}
