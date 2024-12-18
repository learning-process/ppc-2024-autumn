#include "mpi/durynichev_d_custom_allreduce/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

bool durynichev_d_custom_allreduce_mpi::MyAllreduceMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) return taskData->inputs_count[0] > 0;
  return true;
}

bool durynichev_d_custom_allreduce_mpi::MyAllreduceMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* data_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    int data_size = taskData->inputs_count[0];

    data.assign(data_ptr, data_ptr + data_size);
  }

  return true;
}

bool durynichev_d_custom_allreduce_mpi::MyAllreduceMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, data, 0);

  int elements_per_proc = data.size() / world.size();
  int start_index = world.rank() * elements_per_proc;
  int end_index = (world.rank() == world.size() - 1) ? data.size() : start_index + elements_per_proc;

  int local_sum = std::accumulate(data.begin() + start_index, data.begin() + end_index, 0);

  custom_all_reduce(world, local_sum, global_sum, std::plus<>());

  return true;
}

bool durynichev_d_custom_allreduce_mpi::MyAllreduceMPI::post_processing() {
  internal_order_test();

  *reinterpret_cast<int*>(taskData->outputs[0]) = global_sum;

  return true;
}
