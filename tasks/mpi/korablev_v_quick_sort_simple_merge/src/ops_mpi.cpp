#include "mpi/korablev_v_quick_sort_simple_merge/include/ops_mpi.hpp"

#include <mpi.h>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

std::vector<double> korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::merge(
    const std::vector<double>& left, const std::vector<double>& right) {
  std::vector<double> result;
  size_t i = 0;
  size_t j = 0;

  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

std::vector<double> korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::quick_sort_with_merge(
    const std::vector<double>& arr) {
  if (arr.size() <= 1) {
    return std::vector<double>(arr);
  }

  double pivot = arr[arr.size() / 2];
  std::vector<double> left;
  std::vector<double> right;

  for (const auto& elem : arr) {
    if (elem < pivot) {
      left.push_back(elem);
    } else if (elem > pivot) {
      right.push_back(elem);
    }
  }

  std::vector<double> sortedLeft = quick_sort_with_merge(left);
  std::vector<double> sortedRight = quick_sort_with_merge(right);

  std::vector<double> merged = sortedLeft;
  for (const auto& elem : arr) {
    if (elem == pivot) {
      merged.push_back(elem);
    }
  }
  std::vector<double> finalResult = merge(merged, sortedRight);

  return finalResult;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::pre_processing() {
  internal_order_test();

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  auto* input_data = reinterpret_cast<double*>(taskData->inputs[1]);

  input_.assign(input_data, input_data + n);

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 2 || taskData->outputs_count.size() != 1) {
    return false;
  }

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (n < 0 || taskData->inputs_count[1] != n) {
    return false;
  }

  if (taskData->outputs_count[0] != n) {
    return false;
  }

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::run() {
  internal_order_test();

  output_ = quick_sort_with_merge(input_);

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = output_[i];
  }

  return true;
}

std::vector<double> korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::merge(
    const std::vector<double>& left, const std::vector<double>& right) {
  std::vector<double> result;
  size_t i = 0;
  size_t j = 0;

  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

std::vector<double> korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::quick_sort_with_merge(
    const std::vector<double>& arr) {
  if (arr.size() <= 1) {
    return std::vector<double>(arr);
  }

  double pivot = arr[arr.size() / 2];
  std::vector<double> left;
  std::vector<double> right;

  for (const auto& elem : arr) {
    if (elem < pivot) {
      left.push_back(elem);
    } else if (elem > pivot) {
      right.push_back(elem);
    }
  }

  std::vector<double> sortedLeft = quick_sort_with_merge(left);
  std::vector<double> sortedRight = quick_sort_with_merge(right);

  std::vector<double> merged = sortedLeft;
  for (const auto& elem : arr) {
    if (elem == pivot) {
      merged.push_back(elem);
    }
  }
  std::vector<double> finalResult = merge(merged, sortedRight);

  return finalResult;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    auto* input_data = reinterpret_cast<double*>(taskData->inputs[1]);

    input_.assign(input_data, input_data + n);
  }

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 2 || taskData->outputs_count.size() != 1) {
      return false;
    }

    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    if (n < 0 || taskData->inputs_count[1] != n) {
      return false;
    }

    if (taskData->outputs_count[0] != n) {
      return false;
    }
  }

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, input_, 0);

  size_t n = input_.size();

  size_t chunk_size = n / world.size();
  size_t remainder = n % world.size();

  std::vector<int> sizes(world.size(), chunk_size);
  for (size_t i = 0; i < remainder; ++i) {
    sizes[i]++;
  }

  std::vector<int> displs(world.size(), 0);
  for (int i = 1; i < world.size(); ++i) {
    displs[i] = displs[i - 1] + sizes[i - 1];
  }

  local_data_.resize(sizes[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), sizes, displs, local_data_.data(), sizes[world.rank()], 0);

  local_data_ = quick_sort_with_merge(local_data_);

  std::vector<double> gathered_data;
  if (world.rank() == 0) {
    gathered_data.resize(n);
  }
  boost::mpi::gatherv(world, local_data_.data(), sizes[world.rank()], gathered_data.data(), sizes, displs, 0);

  if (world.rank() == 0) {
    output_.assign(gathered_data.begin(), gathered_data.begin() + sizes[0]);
    for (int i = 1; i < world.size(); ++i) {
      std::vector<double> right(gathered_data.begin() + displs[i], gathered_data.begin() + displs[i] + sizes[i]);
      output_ = merge(output_, right);
    }
  }

  return true;
}

bool korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = output_[i];
    }
  }

  return true;
}