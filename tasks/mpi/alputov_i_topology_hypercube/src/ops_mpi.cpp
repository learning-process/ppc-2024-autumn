#include "mpi/alputov_i_topology_hypercube/include/ops_mpi.hpp"

int alputov_i_topology_hypercube_mpi::BinaryToInt(std::vector<int> binary) {
  int result = 0;
  std::reverse(binary.begin(), binary.end());
  for (int i = binary.size() - 1; i >= 0; --i) {
    result += (binary[i] * (1 << i));
  }
  return result;
}

std::vector<int> alputov_i_topology_hypercube_mpi::IntToBinary(int number, int padding) {
  std::vector<int> result;
  int count = 0;
  while (number > 0) {
    result.resize(count + 1);
    result[count++] = number % 2;
    number = number / 2;
  }

  size_t padding_sz = static_cast<size_t>(padding);
  if (padding_sz > result.size()) {
    result.resize(padding_sz);
    for (size_t i = count; i < padding_sz; ++i) result[i] = 0;
  }

  std::reverse(result.begin(), result.end());
  return result;
}

int alputov_i_topology_hypercube_mpi::CalculateNextHop(int sourceRank, int targetRank, int maxAddressBits) {
  std::vector<int> targetBinary = IntToBinary(targetRank, maxAddressBits);
  std::vector<int> sourceBinary = IntToBinary(sourceRank, maxAddressBits);

  for (size_t i = 0; i < targetBinary.size(); ++i) {
    int diff = sourceBinary[i] - targetBinary[i];
    if (diff) {
      sourceBinary[i] = targetBinary[i];
      break;
    }
  }
  return BinaryToInt(sourceBinary);
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    bool isPowerOfTwo = ((world.size() - 1) & world.size()) == 0;
    bool isValidDestination = reinterpret_cast<int *>(taskData->inputs[0])[1] >= 0 &&
                              reinterpret_cast<int *>(taskData->inputs[0])[1] < world.size();

    if (!isPowerOfTwo || !isValidDestination) {
      return false;
    }
  }
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::pre_processing() {
  internal_order_test();

  if (world.rank() != 0) {
    return true;
  }
  routingData.route.clear();
  int *inputData = reinterpret_cast<int *>(taskData->inputs[0]);
  routingData.payload = inputData[0];
  routingData.targetRank = inputData[1];
  routingData.isFinished = false;
  maxAddressBits = IntToBinary(world.size() - 1).size();
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::run() {
  internal_order_test();

  if (world.rank() == 0) {
    routingData.route.resize(1);
    routingData.route[0] = world.rank();

    if (routingData.targetRank == 0) {
      routingData.isFinished = true;
    } else {
      int nextHop = CalculateNextHop(world.rank(), routingData.targetRank, maxAddressBits);
      world.sendrecv(nextHop, 0, routingData, boost::mpi::any_source, 0, routingData);
    }

    for (int i = 0; i < world.size(); ++i) {
      if (std::find(routingData.route.begin(), routingData.route.end(), i) == routingData.route.end()) {
        world.send(i, 0, routingData);
      }
    }
  } else {
    world.recv(boost::mpi::any_source, 0, routingData);

    if (!routingData.isFinished) {
      size_t current_size = routingData.route.size();
      routingData.route.resize(current_size + 1);
      routingData.route[current_size] = world.rank();

      if (world.rank() != routingData.targetRank) {
        world.send(CalculateNextHop(world.rank(), routingData.targetRank, maxAddressBits), 0, routingData);
      } else {
        routingData.isFinished = true;
        world.send(0, 0, routingData);
      }
    }
  }
  return true;
}

bool alputov_i_topology_hypercube_mpi::HypercubeRouterMPI::post_processing() {
  internal_order_test();
  world.barrier();

  if (world.rank() != 0) {
    return true;
  }
  int *outputData = reinterpret_cast<int *>(taskData->outputs[0]);
  outputData[0] = routingData.payload;

  int *outputPath = reinterpret_cast<int *>(taskData->outputs[1]);
  for (size_t i = 0; i < routingData.route.size(); ++i) {
    outputPath[i] = routingData.route[i];
  }

  return true;
}