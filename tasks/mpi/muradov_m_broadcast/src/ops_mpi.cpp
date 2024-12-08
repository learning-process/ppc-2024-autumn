#include "mpi/muradov_m_broadcast/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::validation() {
  internal_order_test();

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
  boost::mpi::broadcast(world, B, source_worker);

  return true;
}

bool muradov_m_broadcast_mpi::BroadcastParallelMPI::post_processing() {
  internal_order_test();

  auto* A_out_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(A.begin(), A.end(), A_out_data);

  auto* B_out_data = reinterpret_cast<int*>(taskData->outputs[1]);
  std::copy(B.begin(), B.end(), B_out_data);

  return true;
}
