#include "mpi/alputov_i_topology_hypercube/include/ops_mpi.hpp"

std::vector<int> alputov_i_topology_hypercube_mpi::IntToBinary(int number, int padding) {
  std::vector<int> result;
  while (number > 0) {
    result.push_back(number % 2);
    number /= 2;
  }
  while (result.size() < (size_t)padding) {
    result.push_back(0);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

int alputov_i_topology_hypercube_mpi::BinaryToInt(std::vector<int> binary) {
  int result = 0;
  std::reverse(binary.begin(), binary.end());
  for (size_t i = 0; i < binary.size(); i++) {
    result += (binary[i] * std::pow(2, i));
  }
  return result;
}

int alputov_i_topology_hypercube_mpi::CalculateNextHop(int sourceRank, int targetRank, int maxAddressBits) {
  std::vector<int> sourceBinary = IntToBinary(sourceRank, maxAddressBits);
  std::vector<int> targetBinary = IntToBinary(targetRank, maxAddressBits);
  for (size_t i = 0; i < targetBinary.size(); i++) {
    if (sourceBinary[i] != targetBinary[i]) {
      sourceBinary[i] = targetBinary[i];
      break;
    }
  }
  return BinaryToInt(sourceBinary);
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (reinterpret_cast<int *>(taskData->inputs[0])[1] >= world.size()) {
      return false;
    }
  }
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    routingData.route.clear();
    routingData.payload = reinterpret_cast<int *>(taskData->inputs[0])[0];
    routingData.targetRank = reinterpret_cast<int *>(taskData->inputs[0])[1];
    routingData.isFinished = false;
    maxAddressBits = IntToBinary(world.size() - 1, 0).size();
  }
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::run() {
  internal_order_test();
  if (world.rank() == 0) {
    routingData.route.push_back(world.rank());
    if (routingData.targetRank != 0) {
      world.send(CalculateNextHop(world.rank(), routingData.targetRank, maxAddressBits), 0, routingData);
      world.recv(boost::mpi::any_source, 0, routingData);
    } else {
      routingData.isFinished = true;
    }
    for (int i = 0; i < world.size(); i++) {
      if (std::find(routingData.route.begin(), routingData.route.end(), i) == routingData.route.end()) {
        world.send(i, 0, routingData);
      }
    }
  } else {
    world.recv(boost::mpi::any_source, 0, routingData);
    if (!routingData.isFinished) {
      routingData.route.push_back(world.rank());
      if (world.rank() == routingData.targetRank) {
        routingData.isFinished = true;
        world.send(0, 0, routingData);
      } else {
        world.send(CalculateNextHop(world.rank(), routingData.targetRank, maxAddressBits), 0, routingData);
      }
    }
  }
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = routingData.payload;
    for (size_t i = 0; i < routingData.route.size(); i++) {
      reinterpret_cast<int *>(taskData->outputs[1])[i] = routingData.route[i];
    }
  }
  return true;
}