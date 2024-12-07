#include "mpi/burykin_m_broadcast/include/ops_mpi.hpp"

bool burykin_m_broadcast_mpi::BroadcastMPI::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != 1) {
    return false;
  }

  int val_source_worker = *taskData->inputs[0];
  return val_source_worker >= 0 && val_source_worker < world.size();
}

bool burykin_m_broadcast_mpi::BroadcastMPI::pre_processing() {
  internal_order_test();

  source_worker = *taskData->inputs[0];

  if (world.rank() == source_worker) {
    auto* input_vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int input_vector_size = taskData->inputs_count[1];

    input_vector.assign(input_vector_data, input_vector_data + input_vector_size);
  }

  return true;
}

bool burykin_m_broadcast_mpi::BroadcastMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, input_vector, source_worker);

  return true;
}

bool burykin_m_broadcast_mpi::BroadcastMPI::post_processing() {
  internal_order_test();

  auto* output_vector_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(input_vector.begin(), input_vector.end(), output_vector_data);

  return true;
}
