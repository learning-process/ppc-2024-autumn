// Copyright 2023 Nesterov Alexander
#include "mpi/suvorov_d_shell_with_ord_merge/include/ops_mpi.hpp"

std::vector<int> suvorov_d_shell_with_ord_merge_mpi::shell_sort(const std::vector<int>& vec_to_sort) {
  std::vector<int> result_vec = vec_to_sort;
  size_t n = result_vec.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; ++i) {
      int curr_val = result_vec[i];
      size_t j = i;

      while (j >= gap && result_vec[j - gap] > curr_val) {
        result_vec[j] = result_vec[j - gap];
        j -= gap;
      }
      result_vec[j] = curr_val;
    }
  }

  return result_vec;
}

void suvorov_d_shell_with_ord_merge_mpi::merge_vectors(const std::vector<int>& left, const std::vector<int>& right,
                                                       std::vector<int>& result) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  result.resize(left.size() + right.size());
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result[k++] = left[i++];
    } else {
      result[k++] = right[j++];
    }
  }

  while (i < left.size()) {
    result[k++] = left[i++];
  }

  while (j < right.size()) {
    result[k++] = right[j++];
  }
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortSeq::pre_processing() {
  internal_order_test();

  auto data_size = static_cast<size_t>(taskData->inputs_count[0]);
  int* data_tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  data_to_sort.assign(data_tmp_ptr, data_tmp_ptr + data_size);

  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortSeq::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs_count.size() == 1 &&
         taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->outputs_count.size() == 1;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortSeq::run() {
  internal_order_test();

  sorted_data = suvorov_d_shell_with_ord_merge_mpi::shell_sort(data_to_sort);

  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortSeq::post_processing() {
  internal_order_test();

  std::copy(sorted_data.begin(), sorted_data.end(), reinterpret_cast<int*>(taskData->outputs[0]));

  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto data_size = static_cast<size_t>(taskData->inputs_count[0]);
    int* data_tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    data_to_sort.assign(data_tmp_ptr, data_tmp_ptr + data_size);
  }

  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] > 0 && taskData->inputs_count.size() == 1 &&
           taskData->outputs_count.size() == 1 && taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel::run() {
  internal_order_test();

  int data_size = 0;
  if (world.rank() == 0) {
    data_size = data_to_sort.size();
  }
  boost::mpi::broadcast(world, data_size, 0);

  int uniform_part_size = data_size / world.size();
  int overflow_size = data_size % world.size();

  std::vector<int> sizes(world.size(), uniform_part_size);

  std::transform(sizes.begin(), sizes.begin() + overflow_size, sizes.begin(), [](int size) { return size + 1; });

  std::vector<int> displacements(world.size(), 0);
  std::partial_sum(sizes.begin(), sizes.end() - 1, displacements.begin() + 1);

  int local_size = sizes[world.rank()];
  partial_data.resize(local_size);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, data_to_sort.data(), sizes, displacements, partial_data.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(world, partial_data.data(), local_size, 0);
  }

  partial_data = suvorov_d_shell_with_ord_merge_mpi::shell_sort(partial_data);

  std::vector<int> merge_data;
  if (world.rank() == 0) {
    merge_data.resize(data_size);
  }
  boost::mpi::gatherv(world, partial_data.data(), sizes[world.rank()], merge_data.data(), sizes, displacements, 0);

  if (world.rank() == 0) {
    std::vector<int> tmp_buff;
    sorted_data.assign(merge_data.begin(), merge_data.begin() + sizes[0]);

    for (int i = 1; i < world.size(); ++i) {
      auto start = merge_data.begin() + displacements[i];
      auto end = start + sizes[i];

      suvorov_d_shell_with_ord_merge_mpi::merge_vectors(sorted_data, std::vector<int>(start, end), tmp_buff);

      sorted_data = std::move(tmp_buff);
    }
  }

  return true;
}

bool suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(sorted_data.begin(), sorted_data.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
