#include "mpi/milovankin_m_sum_of_vector_elements/include/ops_mpi.hpp"

namespace milovankin_m_sum_of_vector_elements_parallel {
//
// Sequential version
//
bool VectorSumSeq::validation() {
  internal_order_test();

  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorSumSeq::pre_processing() {
  internal_order_test();

  // Fill input vector from taskData
  auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());

  return true;
}

bool VectorSumSeq::run() {
  internal_order_test();

  sum_ = 0;
  for (int32_t num : input_) {
    sum_ += num;
  }

  return true;
}

bool VectorSumSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  return true;
}

//
// Parallel version
//

bool VectorSumPar::validation() {
  internal_order_test();

  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorSumPar::pre_processing() {
  internal_order_test();
  sum_ = 0;

  return true;
}

bool VectorSumPar::run() {
  internal_order_test();

  int my_rank = world.rank();
  int world_size = world.size();
  int total_size = 0;

  // Fill input vector from taskData
  if (my_rank == 0) {
    total_size = taskData->inputs_count[0];
    auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_ptr, input_ptr + total_size);
  }

  boost::mpi::broadcast(world, total_size, 0);

  // Create vectors for scatterv
  int local_size = total_size / world_size + (my_rank < (total_size % world_size) ? 1 : 0);
  std::vector<int> send_counts(world_size, total_size / world_size);
  std::vector<int> offsets(world_size, 0);

  // Handle the case when total_size is not divisible by world_size
  for (int i = 0; i < total_size % world_size; ++i) send_counts[i]++;
  for (int i = 1; i < world_size; ++i) offsets[i] = offsets[i - 1] + send_counts[i - 1];

  // Scatter data to local vectors
  local_input_.resize(send_counts[my_rank]);
  boost::mpi::scatterv(world, input_.data(), send_counts, offsets, local_input_.data(), local_size, 0);

  int64_t local_sum = std::accumulate(local_input_.begin(), local_input_.end(), int64_t(0));
  boost::mpi::reduce(world, local_sum, sum_, std::plus<>(), 0);

  return true;
}

bool VectorSumPar::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    *reinterpret_cast<int64_t*>(taskData->outputs[0]) = sum_;
  }

  return true;
}

}  // namespace milovankin_m_sum_of_vector_elements_parallel
