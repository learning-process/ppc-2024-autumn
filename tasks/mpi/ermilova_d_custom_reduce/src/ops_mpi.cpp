// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_custom_reduce/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

bool ermilova_d_custom_reduce_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * cols);

    for (int i = 0; i < rows; i++) {
      auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[i]);
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
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }

  MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

template <typename T>
void apply_operation(void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
  auto *in = reinterpret_cast<T *>(inbuf);
  auto *inout = reinterpret_cast<T *>(inoutbuf);
  for (int i = 0; i < count; i++) {
    if (op == MPI_SUM) {
      inout[i] += in[i];
    }

    else if (op == MPI_MAX)
      inout[i] = (inout[i] > in[i]) ? inout[i] : in[i];
    else if (op == MPI_MIN)
      inout[i] = (inout[i] < in[i]) ? inout[i] : in[i];
    else {
      throw "Unsupported operation\n";
    }
  }
}

int ermilova_d_custom_reduce_mpi::CustomReduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                                               MPI_Op op, int root, MPI_Comm comm) {
  int rank;
  int size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int typesize{};
  MPI_Type_size(datatype, &typesize);
  memcpy(recvbuf, sendbuf, count * typesize);

  int step = 1;
  while (step < size) {
    if (rank % (2 * step) == 0) {
      if (rank + step < size) {
        MPI_Recv(recvbuf, count, datatype, rank + step, 0, comm, MPI_STATUS_IGNORE);
        if (datatype == MPI_INT) {
          apply_operation<int>(recvbuf, sendbuf, count, datatype, op);
        } else if (datatype == MPI_FLOAT) {
          apply_operation<float>(recvbuf, sendbuf, count, datatype, op);
        } else if (datatype == MPI_DOUBLE) {
          apply_operation<double>(recvbuf, sendbuf, count, datatype, op);
        } else {
          fprintf(stderr, "Unsupported datatype\n");
          MPI_Abort(MPI_COMM_WORLD, MPI_ERR_TYPE);
        }
        memcpy(recvbuf, sendbuf, count * typesize);
      }
    } else {
      int dest = rank - step;
      MPI_Send(recvbuf, count, datatype, dest, 0, comm);
      break;
    }
    step *= 2;
  }

  return MPI_SUCCESS;
}
