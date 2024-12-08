// Copyright 2023 Nesterov Alexander
#include <mpi.h>

#include <algorithm>
#include <vector>

#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int start_proc = taskData->inputs_count[0];
  int end_proc = taskData->inputs_count[1];
  int num_of_elems = taskData->inputs_count[2];

  if (start_proc < 0 || end_proc < 0 || num_of_elems <= 0) {
    std::cerr << "Error: Invalid parameters in pre-processing." << std::endl;
    return false;
  }

  if (world.rank() == start_proc) {
    if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      std::cerr << "Error: No input data for the starting process." << std::endl;
      return false;
    }
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
    std::cerr << "Error: Invalid process or element count." << std::endl;
    return false;
  }

  if (world.size() == 1) {
    return true;
  }

  if (start_proc >= end_proc) {
    std::cerr << "Error: start_proc >= end_proc." << std::endl;
    return false;
  }

  if (world.rank() == start_proc) {
    if (taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      std::cerr << "Error: Missing input data for start process." << std::endl;
      return false;
    }
  }

  if (world.rank() == end_proc) {
    if (taskData->outputs.size() < 2 || taskData->outputs[0] == nullptr || taskData->outputs[1] == nullptr) {
      std::cerr << "Error: Missing output data for end process." << std::endl;
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

  int num_requests = 0;

  if (world.rank() == start_proc) {
    if (world.rank() + 1 < world.size()) {
      MPI_Isend(data.data(), num_of_elems, MPI_INT, world.rank() + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
      MPI_Isend(path.data(), path.size(), MPI_INT, world.rank() + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
  } else if (world.rank() == end_proc) {
    if (world.rank() - 1 >= 0) {
      MPI_Irecv(data.data(), num_of_elems, MPI_INT, world.rank() - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
      MPI_Irecv(path.data(), path.size(), MPI_INT, world.rank() - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    path[world.rank() - start_proc] = world.rank();
  } else {
    if (world.rank() - 1 >= 0) {
      MPI_Irecv(data.data(), num_of_elems, MPI_INT, world.rank() - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
      MPI_Irecv(path.data(), path.size(), MPI_INT, world.rank() - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    path[world.rank() - start_proc] = world.rank();

    if (world.rank() + 1 < world.size()) {
      MPI_Isend(data.data(), num_of_elems, MPI_INT, world.rank() + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
      MPI_Isend(path.data(), path.size(), MPI_INT, world.rank() + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
  }

  MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int end_proc = taskData->inputs_count[1];

  if (world.rank() == end_proc) {
    if (taskData->outputs.size() < 2) {
      std::cerr << "Error: Insufficient output space." << std::endl;
      return false;
    }

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
