#include "mpi/muradov_m_broadcast/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

#include "boost/mpi/collectives/reduce.hpp"

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

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::validation() {
  internal_order_test();

  int val_source_worker = *taskData->inputs[0];

  if (world.rank() == val_source_worker) {
    int A_size = taskData->inputs_count[1];
    return A_size > 0;
  }

  return true;
}

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::pre_processing() {
  internal_order_test();

  source_worker = *taskData->inputs[0];

  if (world.rank() == source_worker) {
    auto* A_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int A_size = taskData->inputs_count[1];

    auto* B_data = reinterpret_cast<int*>(taskData->inputs[2]);
    int B_size = taskData->inputs_count[2];

    A.assign(A_data, A_data + A_size);
    B.assign(B_data, B_data + B_size);
  }

  return true;
}

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::run() {
  internal_order_test();

  muradov_m_broadcast_mpi::bcast(world, A, source_worker);
  global_sum_A = calculate_global_sum(world, A, source_worker);

  boost::mpi::broadcast(world, B, source_worker);
  global_sum_B = calculate_global_sum(world, B, source_worker);

  return true;
}

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::post_processing() {
  internal_order_test();

  auto* A_out_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(A.begin(), A.end(), A_out_data);

  auto* B_out_data = reinterpret_cast<int*>(taskData->outputs[1]);
  std::copy(B.begin(), B.end(), B_out_data);

  *reinterpret_cast<int*>(taskData->outputs[2]) = global_sum_A;

  *reinterpret_cast<int*>(taskData->outputs[3]) = global_sum_B;

  return true;
}
