#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <vector>

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int start_proc = static_cast<int>(taskData->inputs_count[0]);
  int end_proc = static_cast<int>(taskData->inputs_count[1]);
  int num_of_elems = static_cast<int>(taskData->inputs_count[2]);

  if (world.rank() == start_proc) {
    const auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    if (input_data != nullptr) {
      data.assign(input_data, input_data + num_of_elems);
    }
    path.clear();
    path.push_back(world.rank());
  }
  if (world.rank() == end_proc && start_proc != end_proc) {
    data.assign(num_of_elems, 0);
    path.assign(world.size(), 0);
  }
  return true;
}

bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) {
    return false;
  }

  int start_proc = static_cast<int>(taskData->inputs_count[0]);
  int end_proc = static_cast<int>(taskData->inputs_count[1]);
  int num_elems = static_cast<int>(taskData->inputs_count[2]);

  if (start_proc < 0 || start_proc >= world.size() || end_proc < 0 || end_proc >= world.size() || num_elems <= 0) {
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

  if (start_proc == end_proc || world.rank() < start_proc || world.rank() > end_proc) {
    return true;
  }

  int data_size = static_cast<int>(taskData->inputs_count[2]);
  int path_size = world.size();

  int* data_arr = new int[data_size];
  int* path_arr = new int[path_size];

  for (size_t i = 0; i < static_cast<size_t>(world.size()); ++i) {
    path_arr[i] = -1;
  }

  if (world.rank() == start_proc) {
    for (size_t i = 0; i < data.size(); ++i) {
      data_arr[i] = data[i];
    }
    path_arr[0] = world.rank();
    boost::mpi::request send_req = world.isend(world.rank() + 1, 0, data_arr, data_size);
    send_req.wait();
    boost::mpi::request send_req1 = world.isend(world.rank() + 1, 1, path_arr, path_size);
    send_req1.wait();
  } else {
    boost::mpi::request recv_req = world.irecv(world.rank() - 1, 0, data_arr, data_size);
    recv_req.wait();
    boost::mpi::request recv_req1 = world.irecv(world.rank() - 1, 1, path_arr, path_size);
    recv_req1.wait();
    path_arr[world.rank()] = world.rank();
    if (world.rank() == end_proc) {
      for (size_t i = 0; i < static_cast<size_t>(data_size); ++i) {
        data[i] = data_arr[i];
      }
      for (size_t i = 0; i < static_cast<size_t>(path_size); ++i) {
        path[i] = path_arr[i];
      }
    }
    if (world.rank() < end_proc) {
      boost::mpi::request send_req = world.isend(world.rank() + 1, 0, data_arr, data_size);
      send_req.wait();
      boost::mpi::request send_req1 = world.isend(world.rank() + 1, 1, path_arr, path_size);
      send_req1.wait();
    }
  }
  delete[] data_arr;
  delete[] path_arr;
  return true;
}
bool lavrentyev_a_line_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int end_proc = static_cast<int>(taskData->inputs_count[1]);

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
