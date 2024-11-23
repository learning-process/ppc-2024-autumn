#include "mpi/gusev_n_cybercube_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
namespace gusev_n_cybercube_topology_mpi {

bool gusev_n_cybercube_topology_mpi::CybercubeTopologyParallel::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    std::cerr << "Validation failed: No input data provided." << std::endl;
    return false;
  }

  int size = world.size();
  if (size < 2) {
    std::cerr << "Validation failed: Insufficient nodes. At least 2 nodes are required." << std::endl;
    return false;
  }

  return true;
}

bool gusev_n_cybercube_topology_mpi::CybercubeTopologyParallel::pre_processing() {
  for (size_t i = 0; i < taskData->inputs.size(); ++i) {
    if (taskData->inputs_count[i] <= 0) {
      std::cerr << "Pre-processing failed: Input size is not valid for input index " << i << std::endl;
      return false;
    }

    if (taskData->inputs[i] == nullptr) {
      std::cerr << "Pre-processing failed: Input pointer is null for input index " << i << std::endl;
      return false;
    }
  }

  world.barrier();
  return true;
}

bool gusev_n_cybercube_topology_mpi::CybercubeTopologyParallel::run() {
  int rank = world.rank();
  int size = world.size();

  // if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
  //   std::cerr << "No input data available for rank " << rank << std::endl;
  //   return false;
  // }

  // if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
  //   std::cerr << "No output data available for rank " << rank << std::endl;
  //   return false;
  // }

  // for (size_t i = 0; i < taskData->outputs_count.size(); ++i) {
  //   /*std::cout << "Output count for rank " << rank << ": " << taskData->outputs_count[i] << std::endl;*/
  //   if (taskData->outputs_count[i] <= 0) {
  //     std::cerr << "Output size is not valid for rank " << rank << std::endl;
  //     return false;
  //   }
  // }

  world.barrier();

  for (int step = 0; step < std::log2(size); ++step) {
    int partner = rank ^ (1 << step);
    if (partner < size) {
      std::vector<uint8_t> send_data(taskData->inputs_count[0]);
      std::copy(taskData->inputs[0], taskData->inputs[0] + taskData->inputs_count[0], send_data.begin());

      /*std::cout << "Rank " << rank << " sending data to " << partner << ": ";
      for (auto val : send_data) std::cout << static_cast<int>(val) << " ";
      std::cout << std::endl;*/

      world.send(partner, 0, send_data);
      std::vector<uint8_t> recv_data(taskData->inputs_count[0]);
      world.recv(partner, 0, recv_data);

      /*std::cout << "Rank " << rank << " received data from " << partner << ": ";
      for (auto val : recv_data) std::cout << static_cast<int>(val) << " ";
      std::cout << std::endl;*/

      if (taskData->outputs_count[0] >= send_data.size()) {
        std::copy(send_data.begin(), send_data.end(), taskData->outputs[0]);
        /*std::cout << "Rank " << rank << " wrote output data: ";
        for (size_t i = 0; i < taskData->outputs_count[0]; ++i) {
          std::cout << static_cast<int>(taskData->outputs[0][i]) << " ";
        }
        std::cout << std::endl;*/
      } else {
        std::cerr << "Output size is not sufficient for rank " << rank << std::endl;
      }
    }
    world.barrier();
  }
  return true;
}

bool gusev_n_cybercube_topology_mpi::CybercubeTopologyParallel::post_processing() {
  /*for (size_t i = 0; i < taskData->outputs.size(); ++i) {
    std::cout << "Output " << i << ": ";
    for (size_t j = 0; j < taskData->outputs_count[i]; ++j) {
      std::cout << static_cast<int>(taskData->outputs[i][j]) << " ";
    }
    std::cout << std::endl;
  }*/
  return true;
}

}  // namespace gusev_n_cybercube_topology_mpi