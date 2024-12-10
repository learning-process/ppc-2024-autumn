#include "mpi/muradov_m_broadcast/include/ops_mpi.hpp"

int calculate_global_sum(boost::mpi::communicator& world, const std::vector<int>& vector, int source_worker) {
  int rank = world.rank();
  int size = world.size();
  int N = vector.size();

  int elements_per_proc = N / size;
  int start_index = rank * elements_per_proc;
  int end_index = (rank == size - 1) ? N : (rank + 1) * elements_per_proc;

  int local_sum = std::accumulate(vector.begin() + start_index, vector.begin() + end_index, 0);

  int global_sum = 0;
  boost::mpi::reduce(world, local_sum, global_sum, std::plus<>(), source_worker);

  return global_sum;
}

bool muradov_m_broadcast_mpi::MyBroadcastParallelMPI::validation() {
  internal_order_test();

  int val_source_worker = *taskData->inputs[0];

  if (world.rank() == val_source_worker) {
    int A_size = taskData->inputs_count[1];
    return A_size > 0;
  }

  return true;
}

bool muradov_m_broadcast_mpi::MyBroadcastParallelMPI::pre_processing() {
  internal_order_test();

  source_worker = *taskData->inputs[0];

  if (world.rank() == source_worker) {
    auto* A_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int A_size = taskData->inputs_count[1];

    A.assign(A_data, A_data + A_size);
  }

  return true;
}

bool muradov_m_broadcast_mpi::MyBroadcastParallelMPI::run() {
  internal_order_test();

  muradov_m_broadcast_mpi::bcast(world, A, source_worker);
  global_sum_A = calculate_global_sum(world, A, source_worker);

  return true;
}

bool muradov_m_broadcast_mpi::MyBroadcastParallelMPI::post_processing() {
  internal_order_test();

  auto* A_out_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(A.begin(), A.end(), A_out_data);

  *reinterpret_cast<int*>(taskData->outputs[1]) = global_sum_A;

  return true;
}

bool muradov_m_broadcast_mpi::MpiBroadcastParallelMPI::validation() {
  internal_order_test();

  int val_source_worker = *taskData->inputs[0];

  if (world.rank() == val_source_worker) {
    int A_size = taskData->inputs_count[1];
    return A_size > 0;
  }

  return true;
}

bool muradov_m_broadcast_mpi::MpiBroadcastParallelMPI::pre_processing() {
  internal_order_test();

  source_worker = *taskData->inputs[0];

  if (world.rank() == source_worker) {
    auto* A_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int A_size = taskData->inputs_count[1];

    A.assign(A_data, A_data + A_size);
  }

  return true;
}

bool muradov_m_broadcast_mpi::MpiBroadcastParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, A, source_worker);
  global_sum_A = calculate_global_sum(world, A, source_worker);

  return true;
}

bool muradov_m_broadcast_mpi::MpiBroadcastParallelMPI::post_processing() {
  internal_order_test();

  auto* A_out_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(A.begin(), A.end(), A_out_data);

  *reinterpret_cast<int*>(taskData->outputs[1]) = global_sum_A;

  return true;
}
