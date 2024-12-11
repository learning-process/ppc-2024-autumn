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
  int s = static_cast<int>(taskData->inputs_count[2]);
  int c = world.size();

  std::vector<boost::mpi::request> req;

  int* d = new int[s];
  int* p = new int[c];

  for (size_t i = 0; i < static_cast<size_t>(world.size()); ++i) {
    p[i] = -1;
  }

  if (start_proc == end_proc) {
    return true;
  }

  if (world.rank() < start_proc || world.rank() > end_proc) {
    return true;
  }

  if (world.rank() == start_proc) {
    for (size_t i = 0; i < data.size(); ++i) {
      d[i] = data[i];
    }
    p[0] = world.rank();
    req.push_back(world.isend(world.rank() + 1, 0, d, s));
    req.push_back(world.isend(world.rank() + 1, 1, p, c));
  } else {
    boost::mpi::request recv_req = world.irecv(world.rank() - 1, 0, d, s);
    recv_req.wait();
    boost::mpi::request recv_req1 = world.irecv(world.rank() - 1, 1, p, c);
    recv_req1.wait();
    p[world.rank()] = world.rank();
    if (world.rank() == end_proc) {
      for (size_t i = 0; i < static_cast<size_t>(s); ++i) {
        data[i] = d[i];
      }
      for (size_t i = 0; i < static_cast<size_t>(c); ++i) {
        path[i] = p[i];
      }
    }
    if (world.rank() < end_proc) {
      req.push_back(world.isend(world.rank() + 1, 0, d, s));
      req.push_back(world.isend(world.rank() + 1, 1, p, c));
    }
  }
  boost::mpi::wait_all(req.begin(), req.end());
  delete[] d;
  delete[] p;
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
