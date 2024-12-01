#include "mpi/tyshkevich_a_hypercube/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::validation() {
  internal_order_test();

  int world_size = world.size();

  int dimension = static_cast<int>(std::log2(world_size));
  if ((1 << dimension) != world_size) {
    return false;
  }

  int val_sender_id = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_target_id = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (val_target_id > val_sender_id) {
    return val_target_id < world_size && val_sender_id >= 0;
  }
  return val_sender_id < world_size && val_target_id >= 0;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::pre_processing() {
  internal_order_test();

  sender_id = *reinterpret_cast<int*>(taskData->inputs[0]);
  target_id = *reinterpret_cast<int*>(taskData->inputs[1]);

  int dimension = static_cast<int>(std::log2(world.size()));
  hypercube = Hypercube(dimension);

  if (world.rank() == sender_id) {
    auto* data = reinterpret_cast<int*>(taskData->inputs[2]);
    int data_size = taskData->inputs_count[2];
    message.assign(data, data + data_size);
  } else if (world.rank() == target_id) {
    int output_data_size = taskData->outputs_count[0];
    result.resize(output_data_size);
  }
  return true;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::run() {
  internal_order_test();

  int world_size = world.size();
  int world_rank = world.rank();

  if (world_rank == sender_id) {
    int next_node = hypercube.getNextNode(world_rank, target_id);
    world.send(next_node, 0, message);
  }

  bool message_received = false;
  while (!message_received) {
    // Слушатель для сообщений с тегом 0
    const auto statOpt0 = world.iprobe(boost::mpi::any_source, 0);
    if (statOpt0.has_value()) {
      std::vector<int> received_data;
      world.recv(boost::mpi::any_source, 0, received_data);

      if (world_rank != target_id) {
        int next_node = hypercube.getNextNode(world_rank, target_id);
        world.send(next_node, 0, received_data);
      } else {
        std::copy(received_data.begin(), received_data.end(), result.begin());
        for (int i = 0; i < world_size; ++i) {
          if (i != target_id) {
            world.send(i, 1, true);  // Termination signal
          }
        }
        message_received = true;
      }
    }

    // Слушатель для сообщений с тегом 1
    const auto statOpt1 = world.iprobe(boost::mpi::any_source, 1);
    if (statOpt1.has_value()) {
      bool terminate;
      world.recv(boost::mpi::any_source, 1, terminate);
      message_received = terminate;
    }
  }
  return true;
}

bool tyshkevich_a_hypercube_mpi::HypercubeParallelMPI::post_processing() {
  internal_order_test();

  boost::mpi::broadcast(world, message, sender_id);

  if (world.rank() == target_id) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), output_data);

    auto* input_data = reinterpret_cast<int*>(taskData->outputs[1]);
    std::copy(message.begin(), message.end(), input_data);
  }

  return true;
}
