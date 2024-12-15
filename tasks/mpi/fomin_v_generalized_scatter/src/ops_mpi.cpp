#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<int> fomin_v_generalized_scatter::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

/*int fomin_v_generalized_scatter::generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                                     MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int left_child = 2 * rank + 1;
  int right_child = 2 * rank + 2;

  if (rank == root) {
    int datatype_size;
    MPI_Type_size(recvtype, &datatype_size);

    const char* send_ptr = reinterpret_cast<const char*>(sendbuf);

    if (left_child < size) {
      int err = MPI_Send(send_ptr + (left_child * recvcount * datatype_size), recvcount, sendtype, left_child, 0, comm);
      if (err != MPI_SUCCESS) {
        //std::cerr << "Error in MPI_Send to process " << left_child << std::endl;
        return err;
      }
    }
    if (right_child < size) {
      int err =
          MPI_Send(send_ptr + (right_child * recvcount * datatype_size), recvcount, sendtype, right_child, 0, comm);
      if (err != MPI_SUCCESS) {
        //std::cerr << "Error in MPI_Send to process " << right_child << std::endl;
        return err;
      }
    }
  } else {
    int err = MPI_Recv(recvbuf, recvcount, recvtype, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
      //std::cerr << "Error in MPI_Recv on process " << rank << std::endl;
      return err;
    }
  }

  return MPI_SUCCESS;
}*/
int fomin_v_generalized_scatter::generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                                     MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int parent = (rank == root) ? MPI_PROC_NULL : (rank - 1) / 2;

  int datatype_size;
  MPI_Type_size(sendtype, &datatype_size);

  if (sendcount != size * recvcount) {
    return MPI_ERR_COUNT;
  }

  if (size == 1) {
    // Single process, copy sendbuf to recvbuf directly
    const char* send_ptr = static_cast<const char*>(sendbuf);
    memcpy(recvbuf, send_ptr, recvcount * datatype_size);
  } else {
    if (rank == root) {
      const char* send_ptr = static_cast<const char*>(sendbuf);
      for (int dest = 0; dest < size; ++dest) {
        if (dest == root) continue;
        int offset = dest * recvcount * datatype_size;
        MPI_Send(send_ptr + offset, recvcount, sendtype, dest, 0, comm);
      }
      int root_offset = root * recvcount * datatype_size;
      memcpy(recvbuf, send_ptr + root_offset, recvcount * datatype_size);
    } else {
      MPI_Recv(recvbuf, recvcount, recvtype, parent, 0, comm, MPI_STATUS_IGNORE);
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
  if (taskData->inputs_count[0] % taskData->outputs_count[0] != 0) {
    // std::cerr << "Error: Input count is not divisible by output count." << std::endl;
    return false;
  }
  // std::cout << "Parallel validation completed." << std::endl;
  return true;
}

bool fomin_v_generalized_scatter::GeneralizedScatterTestParallel::run() {
  internal_order_test();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
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
