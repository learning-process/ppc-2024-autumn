#include "mpi/tyurin_m_shell_sort_batcher_merge/include/ops_mpi.hpp"

namespace tyurin_m_shell_sort_batcher_merge_mpi {

bool ShellSortBatcherMerge::validation() {
  internal_order_test();

  if (world.rank() != 0) return true;

  int val_n = taskData->inputs_count[0];

  return val_n > 0 && val_n % 2 == 0 && world.size() % 2 == 0;
}

bool ShellSortBatcherMerge::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];

  if (world.rank() == 0) {
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_vector.assign(input_ptr, input_ptr + n);
    result.resize(n);
  }

  return true;
}

bool ShellSortBatcherMerge::run() {
  internal_order_test();

  int color = (world.rank() < n) ? 1 : MPI_UNDEFINED;
  boost::mpi::communicator workers = world.split(color);
  if (color == 1) {
    int local_size = n / workers.size();
    std::vector<int> local_data(local_size);
    boost::mpi::scatter(workers, input_vector, local_data.data(), local_size, 0);

    shell_sort_batcher_merge(workers, local_data);

    if (workers.rank() == 0) {
      boost::mpi::gather(workers, local_data.data(), local_data.size(), result.data(), 0);
      merge_sorted_parts(result, workers.size());
    } else {
      boost::mpi::gather(workers, local_data.data(), local_data.size(), 0);
    }
  }

  return true;
}

bool ShellSortBatcherMerge::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), output_ptr);
  }

  return true;
}

}  // namespace tyurin_m_shell_sort_batcher_merge_mpi
