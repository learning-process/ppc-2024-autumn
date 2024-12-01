#include "mpi/baranov_a_qsort_odd_even_mergesort/include/header_b_a_qsort_odd_even_merge.hpp"

namespace baranov_a_qsort_odd_even_merge_mpi {

template <class iotype>
void baranov_a_odd_even_merge_sort<iotype>::merge(std::vector<iotype>& local_data, std::vector<iotype>& other_data) {
  std::vector<iotype> merged(local_data.size() + other_data.size());
  std::merge(local_data.begin(), local_data.end(), other_data.begin(), other_data.end(), merged.begin());

  local_data = merged;
}

template <class iotype>
bool baranov_a_odd_even_merge_sort<iotype>::pre_processing() {
  internal_order_test();
  int myid = world.rank();

  int n;
  if (myid == 0) {
    n = taskData->inputs_count[0];
    input_ = std::vector<iotype>(n);
    output_ = std::vector<iotype>(n);
    void* ptr_r = taskData->inputs[0];
    void* ptr_d = input_.data();
    memcpy(ptr_d, ptr_r, sizeof(iotype) * n);
    vec_size_ = n;
  }
  return true;
}

template <class iotype>
bool baranov_a_odd_even_merge_sort<iotype>::run() {
  internal_order_test();

  int my_rank = world.rank();
  int sz = world.size();
  int n;

  if (sz == 1) {
    output_.assign(input_.begin(), input_.end());
    std::sort(output_.begin(), output_.end());
    return true;
  }

  if (my_rank == 0) {
    int offset = (sz - (input_.size() % sz)) % sz;
    for (int i = 0; i != offset; ++i) {
      input_.push_back(input_.size() + 5);
    }
    n = input_.size();
  }
  broadcast(world, n, 0);

  auto loc_vec_size = n / sz;
  loc_vec_.resize(loc_vec_size);

  boost::mpi::scatter(world, input_, loc_vec_.data(), loc_vec_size, 0);
  std::sort(loc_vec_.begin(), loc_vec_.end());

  bool sz_is_even = (sz % 2 == 0);

  for (int i = 0; i != sz; ++i) {
    world.barrier();
    int low_edge = 0;
    int high_edge = sz;
    if (i % 2 == 0)  // odd iteration
    {
      if (sz_is_even) {
        high_edge = sz;
        low_edge = 0;
      } else {
        low_edge = 0;
        high_edge = sz - 1;
      }
      if (my_rank < low_edge || my_rank >= high_edge) {
        continue;
      }

      int neighbour;
      std::vector<iotype> received_data(loc_vec_size);

      if (my_rank % 2 == 0) {
        neighbour = my_rank + 1;
        world.send(neighbour, 0, loc_vec_);  // even sends to odd

        world.recv(neighbour, 1, received_data);

        merge(loc_vec_, received_data);

        loc_vec_.resize(loc_vec_size);

      } else {
        neighbour = my_rank - 1;
        world.recv(neighbour, 0, received_data);

        world.send(neighbour, 1, loc_vec_);  // odd sends to even

        merge(loc_vec_, received_data);

        auto mid_iter = loc_vec_.begin() + loc_vec_.size() / 2;

        loc_vec_.erase(loc_vec_.begin(), mid_iter);
      }

    } else {  // even iteration
      if (sz_is_even) {
        low_edge = 1;
        high_edge = sz - 1;
      } else {
        low_edge = 1;
        high_edge = sz;
      }

      if (my_rank < low_edge || my_rank >= high_edge) {
        continue;
      }
      int neighbour;
      std::vector<iotype> received_data;
      received_data.reserve(loc_vec_size);
      if (my_rank % 2 != 0) {
        neighbour = my_rank + 1;
        world.send(neighbour, 0, loc_vec_);  // even sends to odd
        world.recv(neighbour, 1, received_data);
        merge(loc_vec_, received_data);

        loc_vec_.resize(loc_vec_size);
      } else {
        neighbour = my_rank - 1;
        world.recv(neighbour, 0, received_data);

        world.send(neighbour, 1, loc_vec_);  // odd sends to even

        merge(loc_vec_, received_data);

        auto mid_iter = loc_vec_.begin() + loc_vec_.size() / 2;

        loc_vec_.erase(loc_vec_.begin(), mid_iter);
      }
    }
  }

  // gather merged
  if (my_rank != 0) {
    boost::mpi::gather(world, loc_vec_.data(), loc_vec_size, 0);
  } else {
    output_.resize(n);
    boost::mpi::gather(world, loc_vec_.data(), loc_vec_size, output_, 0);
  }

  return true;
}
template <class iotype>
bool baranov_a_odd_even_merge_sort<iotype>::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    for (int i = 0; i != vec_size_; ++i) {
      reinterpret_cast<iotype*>(taskData->outputs[0])[i] = output_[i];
    }
    return true;
  }
  return true;
}

template <class iotype>
bool baranov_a_odd_even_merge_sort<iotype>::validation() {
  internal_order_test();
  // check count elements of output
  if (world.rank() == 0) {
    if (taskData->outputs_count[0] == 1 && taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0) {
      return true;
    }
  }
  return true;
}

template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int>;

template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<double>;

template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<unsigned>;
}  // namespace baranov_a_qsort_odd_even_merge_mpi