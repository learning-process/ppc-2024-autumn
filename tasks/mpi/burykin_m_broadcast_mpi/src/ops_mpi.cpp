#include "mpi/burykin_m_broadcast_mpi/include/ops_mpi.hpp"

bool burykin_m_broadcast_mpi_mpi::StdBroadcastMPI::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != 1) return false;

  int val_source_worker = *taskData->inputs[0];

  if (world.size() <= val_source_worker) return false;

  if (val_source_worker < 0 || val_source_worker >= world.size()) {
    return false;
  };

  if (world.rank() == val_source_worker) {
    int input_vector_size = taskData->inputs_count[1];
    if (input_vector_size < 1) {
      return false;
    }
  }

  return true;
}

bool burykin_m_broadcast_mpi_mpi::StdBroadcastMPI::pre_processing() {
  internal_order_test();

  source_worker = *taskData->inputs[0];

  if (world.rank() == source_worker) {
    auto* input_vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int input_vector_size = taskData->inputs_count[1];

    input_vector.assign(input_vector_data, input_vector_data + input_vector_size);
  }

  return true;
}

bool burykin_m_broadcast_mpi_mpi::StdBroadcastMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, input_vector, source_worker);

  int rank = world.rank();
  int size = world.size();
  int N = input_vector.size();

  int elements_per_proc = N / size;
  int remainder = N % size;

  int start_index = rank * elements_per_proc + std::min(rank, remainder);
  int end_index = start_index + elements_per_proc;
  if (rank < remainder) {
    end_index++;
  }

  int local_max = *std::max_element(input_vector.begin() + start_index, input_vector.begin() + end_index);

  reduce(world, local_max, global_max, boost::mpi::maximum<int>(), source_worker);

  return true;
}

bool burykin_m_broadcast_mpi_mpi::StdBroadcastMPI::post_processing() {
  internal_order_test();

  auto* output_vector_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(input_vector.begin(), input_vector.end(), output_vector_data);

  if (world.rank() == source_worker) *reinterpret_cast<int*>(taskData->outputs[1]) = global_max;

  return true;
}
