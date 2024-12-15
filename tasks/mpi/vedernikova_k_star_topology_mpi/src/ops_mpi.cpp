#include "mpi/vedernikova_k_star_topology_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "boost/mpi/communicator.hpp"

constexpr int Term = -1;  // terminating parametr

bool vedernikova_k_star_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() != 0) {
    dest = *reinterpret_cast<int*>(taskData->inputs[0]);
    auto* input_ = reinterpret_cast<int*>(taskData->inputs[1]);
    data.assign(input_, input_ + taskData->inputs_count[1]);
  }

  return true;
}

bool vedernikova_k_star_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  return world.size() > 2 && (world.rank() == 0 || ((taskData->inputs.size() == 2 && taskData->inputs_count[0] == 1)));
}

bool vedernikova_k_star_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  const auto nodes = world.size() - 1;

  if (world.rank() == 0) {
    std::vector<int> buf;
    for (int i = 0; i < nodes; i++) {
      int dst{};
      const auto status = world.recv(boost::mpi::any_source, 0, dst);
      const auto& src = status.source();
      world.recv(src, 0, buf);
      world.send(dst, 0, src);
      world.send(dst, 0, buf);
    }
    for (int i = 0; i < nodes; i++) {
      world.send(i + 1, 0, Term);
    }

  } else {
    world.isend(0, 0, dest);
    world.isend(0, 0, data);
    while (true) {
      int src{};
      world.recv(0, 0, src);
      if (src == Term) {
        break;
      }
      world.recv(0, 0, output);
    }
  }

  return true;
}
bool vedernikova_k_star_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() != 0) {
    auto* _output = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(output.begin(), output.end(), _output);
  }

  return true;
}
