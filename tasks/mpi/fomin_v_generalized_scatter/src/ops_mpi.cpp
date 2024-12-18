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

int fomin_v_generalized_scatter::GeneralizedScatterTestParallel::generalized_scatter(const void* sendbuf, int sendcount,
                                                                                     MPI_Datatype sendtype,
                                                                                     void* recvbuf, int recvcount,
                                                                                     MPI_Datatype recvtype, int root,
                                                                                     MPI_Comm comm) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int datatype_size;
  MPI_Type_size(sendtype, &datatype_size);

  if (sendcount != size * recvcount) {
    return MPI_ERR_COUNT;
  }

  int parent = (rank == root) ? MPI_PROC_NULL : (rank - 1) / 2;
  int left_child = 2 * rank + 1;
  int right_child = 2 * rank + 2;

  if (rank == root) {
    const char* send_ptr = static_cast<const char*>(sendbuf);
    char* recv_ptr = static_cast<char*>(recvbuf);

    for (int child = 0; child < size; ++child) {
      if (child == root) continue;  // Root doesn't send to itself
      int chunk_size = recvcount * datatype_size;
      int offset = child * chunk_size;
      if (child == left_child || child == right_child) {
        MPI_Send(send_ptr + offset, recvcount, sendtype, child, 0, comm);
      }
    }

    // Copy root's chunk to recvbuf
    int root_offset = root * recvcount * datatype_size;
    memcpy(recv_ptr, send_ptr + root_offset, recvcount * datatype_size);
  } else {
    char* recv_ptr = static_cast<char*>(recvbuf);
    MPI_Status status;

    // Receive chunk from parent
    MPI_Recv(recv_ptr, recvcount, recvtype, parent, 0, comm, &status);

    // Send chunk to children if any
    if (left_child < size) {
      MPI_Send(recv_ptr, recvcount, recvtype, left_child, 0, comm);
    }
    if (right_child < size) {
      MPI_Send(recv_ptr, recvcount, recvtype, right_child, 0, comm);
    }
  }

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