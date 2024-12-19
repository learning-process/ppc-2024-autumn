#include "mpi/gusev_n_hypercube_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <numeric>
#include <vector>
namespace gusev_n_hypercube_topology_mpi {

bool gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel::validation() {
  // if (!((std::ceil(std::log2(world.size())) == std::log2(world.size())) && (world.size() >= 2))) {
  //   /*std::cerr << "Warning: world size != 2^n" << std::endl;*/
  // }

  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    /*std::cerr << "Validation failed: No input data provided." << std::endl;*/
    return false;
  }

  // int size = world.size();
  // if (size < 2) {
  //   /*std::cerr << "Validation failed: Insufficient nodes. At least 2 nodes are required." << std::endl;*/
  //   return false;
  // }

  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0) {
      /*std::cerr << "Pre-processing failed: Input size is not valid for input index " << i << std::endl;*/
      return false;
    }

    if (taskData->inputs[i] == nullptr) {
      /*std::cerr << "Pre-processing failed: Input pointer is null for input index " << i << std::endl;*/
      return false;
    }
  }

  return true;
}

bool gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel::pre_processing() { return true; }

bool gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();

  world.barrier();

  // world size < 2
  if (size < 2) {
    if (taskData->outputs_count[0] >= taskData->inputs_count[0]) {
      std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], taskData->outputs[0]);
    }
    return true;
  }

  for (int step = 0; step < std::log2(size); ++step) {
    int partner = rank ^ (1 << step);
    if (partner < size) {
      std::vector<uint8_t> send_data(taskData->inputs_count[0]);
      std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

      world.send(partner, 0, send_data);
      std::vector<uint8_t> recv_data(taskData->inputs_count[0]);
      world.recv(partner, 0, recv_data);

      if (taskData->outputs_count[0] >= send_data.size()) {
        std::copy(send_data.begin(), send_data.end(), taskData->outputs[0]);
      } /*else {
        std::cerr << "Output size is not sufficient for rank " << rank << std::endl;
      }*/
    }
    world.barrier();
  }
  return true;
}

bool gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel::post_processing() { return true; }

}  // namespace gusev_n_hypercube_topology_mpi
