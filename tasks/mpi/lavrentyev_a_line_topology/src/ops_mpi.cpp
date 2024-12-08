// Copyright 2023 Nesterov Alexander
#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int start_proc = taskData->inputs_count[0];
  int end_proc = taskData->inputs_count[1];
  int num_of_elems = taskData->inputs_count[2];

  if (world.rank() == start_proc) {
    const auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    data.assign(input_data, input_data + num_of_elems);
    path.clear();
    path.push_back(world.rank());
  } else {
    data.resize(num_of_elems, 0);
    int path_size = end_proc - start_proc + 1;
    path.resize(path_size, -1);
  }

  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) {
    std::cerr << "Error: inputs_count has less than 3 elements." << std::endl;
    return false;
  }

  int start_proc = taskData->inputs_count[0];
  int end_proc = taskData->inputs_count[1];
  int num_of_elems = taskData->inputs_count[2];

  if (start_proc < 0 || start_proc >= world.size() || end_proc < 0 || end_proc >= world.size() || num_of_elems <= 0) {
    return false;
  }

  if (world.size() == 1) {
    return true;
  }

  if (start_proc >= end_proc) {
    return false;
  }

  if (world.rank() == start_proc) {
    if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      return false;
    }
  }

  if (world.rank() == end_proc) {
    if (taskData->outputs.empty() || taskData->outputs[0] == nullptr || taskData->outputs[1] == nullptr) {
      return false;
    }
  }

  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int start_proc = taskData->inputs_count[0];
  int end_proc = taskData->inputs_count[1];
  int num_of_elems = taskData->inputs_count[2];

  if (world.rank() < start_proc || world.rank() > end_proc) {
    return true;
  }

  MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

  if (world.rank() == start_proc) {
    MPI_Isend(data.data(), num_of_elems, MPI_INT, world.rank() + 1, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(path.data(), path.size(), MPI_INT, world.rank() + 1, 1, MPI_COMM_WORLD, &requests[1]);
  } else if (world.rank() == end_proc) {
    MPI_Irecv(data.data(), num_of_elems, MPI_INT, world.rank() - 1, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(path.data(), path.size(), MPI_INT, world.rank() - 1, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    path[world.rank() - start_proc] = world.rank();
  } else {
    MPI_Irecv(data.data(), num_of_elems, MPI_INT, world.rank() - 1, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(path.data(), path.size(), MPI_INT, world.rank() - 1, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    path[world.rank() - start_proc] = world.rank();

    MPI_Isend(data.data(), num_of_elems, MPI_INT, world.rank() + 1, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(path.data(), path.size(), MPI_INT, world.rank() + 1, 1, MPI_COMM_WORLD, &requests[3]);
  }

  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int end_proc = taskData->inputs_count[1];

  if (world.rank() == end_proc) {
    auto* data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    if (data_ptr != nullptr) {
      std::copy(data.begin(), data.end(), data_ptr);
    }
    auto* path_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
    if (path_ptr != nullptr) {
      std::copy(path.begin(), path.end(), path_ptr);
    }
  }

  return true;
}
