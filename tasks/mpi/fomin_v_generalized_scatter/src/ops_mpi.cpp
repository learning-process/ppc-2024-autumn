#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct TaskData {
  void* inputs[1];
  int inputs_count[1];
  void* outputs[1];
  int outputs_count[1];
  MPI_Datatype datatype;
  std::string ops;
};

namespace fomin_v_generalized_scatter {

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

int generalized_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                        MPI_Datatype recvtype, int root, MPI_Comm comm) {
  internal_order_test();
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int parent = (rank - 1) / 2;
  int left_child = 2 * rank + 1;
  int right_child = 2 * rank + 2;

  if (rank == root) {
    for (int child = left_child; child < size; child = right_child + 1) {
      if (child >= size) break;
      int err = MPI_Send(sendbuf + (child * recvcount * MPI_Type_size(recvtype)), recvcount, sendtype, child, 0, comm);
      if (err != MPI_SUCCESS) {
        std::cerr << "Error in MPI_Send from root to process " << child << std::endl;
        return err;
      }
    }
  } else {
    int err = MPI_Recv(recvbuf, recvcount, recvtype, parent, 0, comm, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) {
      std::cerr << "Error in MPI_Recv on process " << rank << " from parent " << parent << std::endl;
      return err;
    }
  }

  return MPI_SUCCESS;
}

class GeneralizedScatterTestSequential {
 public:
  bool post_processing(TaskData* taskData) {
    internal_order_test();

    std::cout << "Sequential post-processing completed." << std::endl;
    return true;
  }
};

class GeneralizedScatterTestParallel {
 public:
  bool pre_processing(TaskData* taskData) {
    internal_order_test();
    std::cout << "Parallel pre-processing completed." << std::endl;
    return true;
  }

  bool validation(TaskData* taskData) {
    internal_order_test();
    if (taskData->inputs_count[0] % taskData->outputs_count[0] != 0) {
      std::cerr << "Error: Input count is not divisible by output count." << std::endl;
      return false;
    }
    std::cout << "Parallel validation completed." << std::endl;
    return true;
  }

  bool run(TaskData* taskData) {
    internal_order_test();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int root = 0;

    int sendcount = taskData->inputs_count[0];
    int recvcount = sendcount / size;

    if (rank == root) {
      int err = generalized_scatter(taskData->inputs[0], sendcount, taskData->datatype, taskData->outputs[0], recvcount,
                                    taskData->datatype, root, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS) {
        std::cerr << "Error in generalized_scatter on root process." << std::endl;
        return false;
      }
    } else {
      int err = generalized_scatter(nullptr, 0, taskData->datatype, taskData->outputs[0], recvcount, taskData->datatype,
                                    root, MPI_COMM_WORLD);
      if (err != MPI_SUCCESS) {
        std::cerr << "Error in generalized_scatter on process " << rank << std::endl;
        return false;
      }
    }

    return true;
  }

  bool post_processing(TaskData* taskData) {
    internal_order_test();

    std::cout << "Parallel post-processing completed." << std::endl;
    return true;
  }
};

}  // namespace fomin_v_generalized_scatter