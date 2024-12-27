#include "mpi/sarafanov_m_quick_sort_batcher_merge/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

namespace sarafanov_m_quick_sort_batcher_merge_mpi {

void make_bitonic_sequence(std::vector<int>& arr) {
  size_t n = arr.size();
  quickSort(arr.begin(), arr.begin() + n / 2);
  quickSort(arr.begin() + n / 2, arr.end(), std::greater<>());
}

void merge_batcher(boost::mpi::communicator& world, std::vector<int>& arr, int arr_size) {
  int world_size = world.size();
  int world_rank = world.rank();

  int max_power_of_two = 1;
  while (max_power_of_two * 2 <= world_size) {
    max_power_of_two *= 2;
  }

  int color = (world_rank < max_power_of_two) ? 0 : MPI_UNDEFINED;
  boost::mpi::communicator sub_comm = world.split(color);

  if (color == 0) {
    if (sub_comm.rank() == 0) make_bitonic_sequence(arr);

    std::vector<int> local_arr(arr_size / sub_comm.size());
    boost::mpi::scatter(sub_comm, arr.data(), local_arr.data(), arr_size / sub_comm.size(), 0);

    std::vector<int> iter_send;

    for (int i = sub_comm.size() / 2; i > 0; i /= 2) {
      iter_send.clear();
      if (sub_comm.rank() == 0) {
        for (int rank = 0; rank < sub_comm.size(); rank++) {
          if (rank - i < 0 || (rank + i < sub_comm.size() &&
                               std::find(iter_send.begin(), iter_send.end(), rank - i) == iter_send.end())) {
            iter_send.push_back(rank);
          }
        }
      }
      boost::mpi::broadcast(sub_comm, iter_send, 0);

      int iter_color = (std::find(iter_send.begin(), iter_send.end(), sub_comm.rank()) != iter_send.end()) ? 0 : 1;
      if (iter_color == 0) {
        world.send(sub_comm.rank() + i, 0, local_arr);
        world.recv(sub_comm.rank() + i, 1, local_arr);
      } else {
        std::vector<int> recv_data(local_arr.size());
        world.recv(sub_comm.rank() - i, 0, recv_data);

        for (size_t j = 0; j < local_arr.size(); ++j) {
          if (recv_data[j] > local_arr[j]) {
            std::swap(recv_data[j], local_arr[j]);
          }
        }

        world.send(sub_comm.rank() - i, 1, recv_data);
      }
    }
    quickSort(local_arr.begin(), local_arr.end());
    boost::mpi::gather(sub_comm, local_arr.data(), arr_size / sub_comm.size(), arr.data(), 0);
  }
}

bool QuicksortBatcherMerge::validation() {
  internal_order_test();

  if (world.rank() != 0) return true;

  int val_arr_size = taskData->inputs_count[0];
  int val_out_arr_size = taskData->outputs_count[0];

  return val_arr_size > 0 && val_out_arr_size == val_arr_size && (val_arr_size & (val_arr_size - 1)) == 0;
}

bool QuicksortBatcherMerge::pre_processing() {
  internal_order_test();

  vector_size = *reinterpret_cast<int*>(taskData->inputs[0]);

  if (world.rank() == 0) {
    auto* vec_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vec_size = taskData->inputs_count[1];

    arr.assign(vec_data, vec_data + vec_size);
  }

  return true;
}

bool QuicksortBatcherMerge::run() {
  internal_order_test();

  merge_batcher(world, arr, vector_size);

  return true;
}

bool QuicksortBatcherMerge::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out_vector = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(arr.begin(), arr.end(), out_vector);
  }

  return true;
}

}  // namespace sarafanov_m_quick_sort_batcher_merge_mpi
