#include "mpi/tyshkevich_a_hypercube/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <utility>
#include <vector>

void printPath(const std::vector<int>& path) {
  for (size_t i = 0; i < path.size(); ++i) {
    std::cout << path[i];
    if (i != path.size() - 1) {
      std::cout << " -> ";
    }
  }
  std::cout << std::endl;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::validation() {
  internal_order_test();

  int world_size = world.size();

  int val_dimension = static_cast<int>(std::log2(world_size));
  if ((1 << val_dimension) != world_size) {
    return false;
  }

  int val_sender_id = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_target_id = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (val_target_id == val_sender_id) {
    return false;
  }
  if (val_target_id > val_sender_id) {
    return val_target_id < world_size && val_sender_id >= 0;
  }
  return val_sender_id < world_size && val_target_id >= 0;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::pre_processing() {
  internal_order_test();

  int sender = *reinterpret_cast<int*>(taskData->inputs[0]);
  int target = *reinterpret_cast<int*>(taskData->inputs[1]);

  sender_id = sender;
  target_id = target;

  dimension = static_cast<int>(std::log2(world.size()));

  if (world.rank() == sender_id) {
    auto* data = reinterpret_cast<int*>(taskData->inputs[2]);
    int data_size = taskData->inputs_count[2];
    std::vector<int> data_input(data_size);
    data_input.assign(data, data + data_size);
    message.resize(data_input.size());
    std::copy(data_input.begin(), data_input.end(), message.begin());
  }

  int current = sender_id;
  do {
    shortest_route.push_back(current);
    current = tyshkevich_a_hypercube_mpi::getNextNode(current, target_id, dimension);
  } while (current != target_id);

  shortest_route.push_back(target_id);

  return true;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::run() {
  internal_order_test();

  int world_rank = world.rank();

  if (world_rank == sender_id) {
    int next_node = shortest_route[1];
    auto send_pair = std::make_pair(message, 1);
    world.send(next_node, 0, send_pair);
  } else {
    if (std::find(shortest_route.begin(), shortest_route.end(), world_rank) != shortest_route.end()) {
      std::pair<std::vector<int>, int> received_data;
      world.recv(boost::mpi::any_source, 0, received_data);

      if (world_rank != target_id) {
        int next_node = shortest_route[++received_data.second];
        world.send(next_node, 0, received_data);
      } else {
        result.resize(received_data.first.size());
        std::copy(received_data.first.begin(), received_data.first.end(), result.begin());
        route_iters = received_data.second + 1;
      }
    }
  }

  return true;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::post_processing() {
  internal_order_test();

  world.barrier();
  if (world.rank() == target_id) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), output_data);

    *reinterpret_cast<int*>(taskData->outputs[1]) = route_iters;
  }

  return true;
}
