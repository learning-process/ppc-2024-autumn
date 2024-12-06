#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"

#include <limits>

#include "boost/mpi/operations.hpp"
// #include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"
/*
template <typename T>
std::vector<T> agafeev_s_max_of_vector_elements_mpi::create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

bool agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq::pre_processing() {
  internal_order_test();

  // Init value
  auto* temp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);

  return true;
}

bool agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq::run() {
  internal_order_test();

  maxres_ = get_MaxValue(input_);

  return true;
}

bool agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = maxres_;

  return true;
}
*/
// Parallel
namespace agafeev_s_max_of_vector_elements_mpi {

template <typename T>
bool MaxMatrixMpi<T>::pre_processing() {
  internal_order_test();
  maxres_ = std::numeric_limits<T>::min();

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) return (taskData->outputs_count[0] == 1 && !(taskData->inputs.empty()));

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::run() {
  internal_order_test();

  unsigned int world_rank = world.rank();
  unsigned int world_size = world.size();
  unsigned int data_size = 0;

  if (world_rank == 0) {
    data_size = taskData->inputs_count[0];
    auto* temp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);
  }

  boost::mpi::broadcast(world, data_size, 0);

  unsigned int task_size = data_size / world_size;
  unsigned int over_size = data_size % world_size;

  std::vector<int> sizes(world_size, task_size);
  std::vector<int> displs(world_size, 0);

  if (world_rank == 0) {
    for (unsigned int i = 0; i < over_size; ++i) sizes[i]++;
    for (unsigned int i = 1; i < world_size; ++i) displs[i] = displs[i - 1] + sizes[i - 1];
  }

  local_vector.resize(sizes[world_rank]);

  boost::mpi::scatterv(world, input_.data(), sizes, displs, local_vector.data(), sizes[world_rank], 0);
  T res = agafeev_s_max_of_vector_elements_mpi::get_MaxValue<T>(local_vector);
  boost::mpi::reduce(world, res, maxres_, boost::mpi::maximum<T>(), 0);

  return true;
}

template <typename T>
bool MaxMatrixMpi<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) reinterpret_cast<T*>(taskData->outputs[0])[0] = maxres_;

  return true;
}

template class MaxMatrixMpi<int>;
}  // namespace agafeev_s_max_of_vector_elements_mpi
