#include "mpi/korotin_e_my_scatter/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int korotin_e_my_scatter_mpi::TestMPITaskMyParallel::MPI_My_Scatter(void* send_buf, int sendcount,
                                                                    MPI_Datatype sendtype, void* recv_buf,
                                                                    int recvcount, MPI_Datatype recvtype, int src,
                                                                    MPI_Comm comm) {
  int worldsize;
  int size;
  int rank;
  int current = 0;
  int parent = -1;
  int weight;
  int left = -1;
  int right = -1;
  int type_size;
  char* sendbuf = reinterpret_cast<char*>(send_buf);
  char* recvbuf = reinterpret_cast<char*>(recv_buf);

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &worldsize);
  size = worldsize;
  weight = worldsize;

  MPI_Type_size(sendtype, &type_size);

  while (rank != current) {
    parent = current;
    weight = (weight - 1) / 2;
    if (rank > weight + current) {
      size /= 2;
      current += weight;
    } else
      size = weight;
    current++;
  }

  if (size > 2) {
    left = current + 1;
    right = left + (size - 1) / 2;
  } else if (size == 2) {
    right = current + 1;
  }

  if (rank == src) {
    if (src != 0) {
      if (src != 1) MPI_Send(sendbuf + sendcount * type_size, ((worldsize - 1) / 2) * sendcount, sendtype, 1, 0, comm);
      if (src != worldsize - worldsize / 2)
        MPI_Send(sendbuf + (worldsize - worldsize / 2) * sendcount * type_size, (worldsize / 2) * sendcount, sendtype,
                 worldsize - (worldsize / 2), 0, comm);
      MPI_Send(sendbuf, sendcount, sendtype, 0, 0, comm);
    }
    if (left > -1) {
      MPI_Send(sendbuf + left * sendcount * type_size, ((size - 1) / 2) * sendcount, sendtype, left, 0, comm);
    }
    if (right > -1) {
      MPI_Send(sendbuf + right * sendcount * type_size, (size / 2) * sendcount, sendtype, right, 0, comm);
    }

    std::copy(sendbuf + rank * sendcount * type_size, sendbuf + (rank + 1) * sendcount * type_size, recvbuf);
  } else if (rank != 0) {
    char* tmpbuf = new char[size * recvcount * type_size];

    if (parent == 0) parent = src;
    MPI_Recv(tmpbuf, recvcount * size, recvtype, parent, 0, comm, MPI_STATUS_IGNORE);

    if (left > -1 && left != src) {
      MPI_Send(tmpbuf + sendcount * type_size, ((size - 1) / 2) * sendcount, sendtype, left, 0, comm);
    }
    if (right > -1 && right != src) {
      MPI_Send(tmpbuf + (right - rank) * sendcount * type_size, (size / 2) * sendcount, sendtype, right, 0, comm);
    }

    std::copy(tmpbuf, tmpbuf + sendcount * type_size, recvbuf);
    delete[] tmpbuf;
  } else
    MPI_Recv(recvbuf, recvcount, recvtype, src, 0, comm, MPI_STATUS_IGNORE);
  return MPI_SUCCESS;
}

bool korotin_e_my_scatter_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(start, start + taskData->inputs_count[0], input_.begin());
  res = 0.0;
  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_my_scatter_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = *std::max_element(input_.begin(), input_.end());
  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskMyParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == root) {
    input_ = std::vector<double>(taskData->inputs_count[0]);
    auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(start, start + taskData->inputs_count[0], input_.begin());
  }
  res = 0.0;
  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskMyParallel::validation() {
  internal_order_test();
  if (world.rank() == root) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskMyParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == root) {
    delta = taskData->inputs_count[0] / world.size();
    local_input_ = std::vector<double>(delta + taskData->inputs_count[0] % world.size());
    std::copy(input_.begin() + delta * world.size(), input_.end(), local_input_.begin() + delta);
  }

  broadcast(world, delta, root);

  if (world.rank() != root) {
    local_input_ = std::vector<double>(delta);
  }

  MPI_My_Scatter(input_.data(), delta, MPI_DOUBLE, local_input_.data(), delta, MPI_DOUBLE, root, MPI_COMM_WORLD);

  double local_res;

  if (local_input_.empty())
    local_res = -INFINITY;
  else
    local_res = *std::max_element(local_input_.begin(), local_input_.end());

  reduce(world, local_res, res, boost::mpi::maximum<double>(), root);

  return true;
}

bool korotin_e_my_scatter_mpi::TestMPITaskMyParallel::post_processing() {
  internal_order_test();
  if (world.rank() == root) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
