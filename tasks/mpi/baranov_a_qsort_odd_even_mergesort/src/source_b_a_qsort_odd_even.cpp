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

  if (myid == 0) {
    vec_size_ = taskData->inputs_count[0];
    input_ = std::vector<iotype>(vec_size_);
    output_ = std::vector<iotype>(vec_size_);
    void* ptr_r = taskData->inputs[0];
    void* ptr_d = input_.data();
    memcpy(ptr_d, ptr_r, sizeof(iotype) * vec_size_);
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
    output_ = input_;
    output_ = q_sort_stack(output_);
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

  loc_vec_ = q_sort_stack(loc_vec_);

  bool sz_is_even = (sz % 2 == 0);

  for (int i = 0; i != sz; ++i) {
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
    if (taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 2) {
      return true;
    }
  }
  return true;
}

template <typename iotype>
std::vector<iotype> baranov_a_odd_even_merge_sort<iotype>::q_sort_stack(std::vector<iotype>& vec_) {
  if (vec_.empty()) {
    return vec_;
  }

  struct Range {
    int low;
    int high;
  };

  std::stack<Range> ranges;
  ranges.push({0, static_cast<int>(vec_.size() - 1)});
  std::vector<iotype> result = vec_;
  while (!ranges.empty()) {
    Range range = ranges.top();
    ranges.pop();

    int low = range.low;
    int high = range.high;

    if (high - low <= 10) {
      std::sort(result.begin() + low, result.begin() + high + 1);
      continue;
    }

    iotype pivot = result[high];
    std::vector<iotype> left;
    std::vector<iotype> right;
    std::vector<iotype> equal;

    for (int i = low; i <= high; ++i) {
      if (result[i] < pivot) {
        left.push_back(result[i]);
      } else if (result[i] > pivot) {
        right.push_back(result[i]);
      } else {
        equal.push_back(result[i]);
      }
    }

    std::vector<iotype> merged_left;
    std::merge(left.begin(), left.end(), equal.begin(), equal.end(), std::back_inserter(merged_left));
    std::merge(merged_left.begin(), merged_left.end(), right.begin(), right.end(), result.begin() + low);

    if (!left.empty()) {
      ranges.push({low, low + static_cast<int>(left.size()) - 1});
    }
    if (!right.empty()) {
      ranges.push({low + static_cast<int>(merged_left.size()), high});
    }
  }

  return result;
}

template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<int>;

template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<double>;
template class baranov_a_qsort_odd_even_merge_mpi::baranov_a_odd_even_merge_sort<unsigned int>;

}  // namespace baranov_a_qsort_odd_even_merge_mpi