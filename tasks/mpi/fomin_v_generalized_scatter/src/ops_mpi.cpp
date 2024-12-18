#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> fomin_v_generalized_scatter::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

int fomin_v_generalized_scatter::generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                                     MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int datatype_size;
  MPI_Type_size(sendtype, &datatype_size);

  if (rank == root && sendcount != size * recvcount) {
    return MPI_ERR_COUNT;
  }

  int parent = (rank == root) ? MPI_PROC_NULL : (rank - 1) / 2;
  int left_child = 2 * rank + 1;
  int right_child = 2 * rank + 2;

  // Buffer for temporary storage in non-root processes
  char* temp_buffer = (rank == root) ? nullptr : new char[sendcount * datatype_size];

  if (rank == root) {
    // Copy the root's data to its recvbuf
    const char* send_ptr = static_cast<const char*>(sendbuf);
    memcpy(recvbuf, send_ptr + rank * recvcount * datatype_size, recvcount * datatype_size);

    // Send data to children
    if (left_child < size) {
      MPI_Send(send_ptr + left_child * recvcount * datatype_size, (size - left_child) * recvcount * datatype_size,
               sendtype, left_child, 0, comm);
    }
    if (right_child < size) {
      MPI_Send(send_ptr + right_child * recvcount * datatype_size, (size - right_child) * recvcount * datatype_size,
               sendtype, right_child, 0, comm);
    }
  } else {
    // Receive data from parent
    MPI_Status status;
    MPI_Recv(temp_buffer, sendcount * datatype_size, sendtype, parent, 0, comm, &status);

    // Copy the rank's data to its recvbuf
    memcpy(recvbuf, temp_buffer + rank * recvcount * datatype_size, recvcount * datatype_size);

    // Forward data to children
    if (left_child < size) {
      MPI_Send(temp_buffer + left_child * recvcount * datatype_size, (size - left_child) * recvcount * datatype_size,
               sendtype, left_child, 0, comm);
    }
    if (right_child < size) {
      MPI_Send(temp_buffer + right_child * recvcount * datatype_size, (size - right_child) * recvcount * datatype_size,
               sendtype, right_child, 0, comm);
    }
  }

  delete[] temp_buffer;
  return MPI_SUCCESS;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::pre_processing() {
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[0] % taskData->outputs_count[0] == 0;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::run() {
  internal_order_test();
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int root = 0;

  int sendcount = taskData->inputs_count[0];
  int recvcount = sendcount / size;

  if (rank == root) {
    int err = generalized_scatter(taskData->inputs[0], sendcount, MPI_INT, taskData->outputs[0], recvcount, MPI_INT,
                                  root, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      // std::cerr << "Error in generalized_scatter on root process." << std::endl;
      return false;
    }
  } else {
    int err = generalized_scatter(nullptr, 0, MPI_INT, taskData->outputs[0], recvcount, MPI_INT, root, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
      // std::cerr << "Error in generalized_scatter on process " << rank << std::endl;
      return false;
    }
  }

  return true;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}