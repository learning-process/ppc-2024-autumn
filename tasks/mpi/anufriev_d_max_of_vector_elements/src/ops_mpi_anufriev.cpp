// mpi/src/ops_mpi.cpp
#include <limits>
#include <numeric>
#include <random>

#include "mpi/anufriev_d_max_of_vector_elements/include/ops_mpi_anufriev.hpp"

namespace anufriev_d_max_of_vector_elements_parallel {

std::vector<int32_t> make_random_vector(int32_t size, int32_t val_min, int32_t val_max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(val_min, val_max);

  std::vector<int32_t> new_vector(size);
  std::generate(new_vector.begin(), new_vector.end(), [&]() { return distrib(gen); });
  return new_vector;
}

// Sequential Version
bool VectorMaxSeq::validation() {
  internal_order_test();
  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorMaxSeq::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(taskData->inputs_count[0]);
  std::copy(input_ptr, input_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool VectorMaxSeq::run() {
  internal_order_test();
  if (input_.empty()) {
    return true;
  }
  max_ = input_[0];
  for (int32_t num : input_) {
    if (num > max_) {
      max_ = num;
    }
  }
  return true;
}

bool VectorMaxSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int32_t*>(taskData->outputs[0]) = max_;
  return true;
}

// Parallel Version
bool VectorMaxPar::validation() {
  internal_order_test();
  return !taskData->outputs.empty() && taskData->outputs_count[0] == 1;
}

bool VectorMaxPar::pre_processing() {
  internal_order_test();
  max_ = std::numeric_limits<int32_t>::min();
  return true;
}

bool VectorMaxPar::run() {
  internal_order_test();

  int my_rank = world.rank();
  int world_size = world.size();
  int total_size = 0;

  if (my_rank == 0) {
    total_size = taskData->inputs_count[0];
    auto* input_ptr = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_ptr, input_ptr + total_size);
  }

  boost::mpi::broadcast(world, total_size, 0);

  int local_size = total_size / world_size + (my_rank < (total_size % world_size) ? 1 : 0);
  std::vector<int> send_counts(world_size, total_size / world_size);
  std::vector<int> offsets(world_size, 0);

  for (int i = 0; i < total_size % world_size; ++i) {
    send_counts[i]++;
  }
  for (int i = 1; i < world_size; ++i) {
    offsets[i] = offsets[i - 1] + send_counts[i - 1];
  }

  local_input_.resize(send_counts[my_rank]);
  boost::mpi::scatterv(world, input_.data(), send_counts, offsets, local_input_.data(), local_size, 0);

  int32_t local_max = std::numeric_limits<int32_t>::min();
  for (int32_t num : local_input_) {
    if (num > local_max) {
      local_max = num;
    }
  }
  boost::mpi::reduce(world, local_max, max_, boost::mpi::maximum<int32_t>(), 0);

  return true;
}

bool VectorMaxPar::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int32_t*>(taskData->outputs[0]) = max_;
  }
  return true;
}

}  // namespace anufriev_d_max_of_vector_elements_parallel