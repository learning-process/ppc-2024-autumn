#include "mpi/rams_s_radix_sort_with_simple_merge_for_doubles/include/ops_mpi.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>

bool rams_s_radix_sort_with_simple_merge_for_doubles_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto *input_data = reinterpret_cast<double *>(taskData->inputs[0]);
    input = std::vector<double>(input_data, input_data + taskData->inputs_count[0]);
    result = std::vector<double>(taskData->outputs_count[0]);
  }
  return true;
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return world.rank() != 0 ||
         (taskData->inputs_count[0] >= 0 && taskData->inputs_count[0] == taskData->outputs_count[0]);
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t world_size = world.size();
  size_t input_length;
  if (world.rank() == 0) {
    input_length = input.size();
  }
  boost::mpi::broadcast(world, input_length, 0);

  size_t avg_to_send = input_length / world.size();
  size_t extra_to_send = input_length % world.size();
  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size, 0);
  for (size_t i = 0; i < world_size; i++) {
    sendcounts[i] = avg_to_send + (i < extra_to_send ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  auto local_input = std::vector<double>(sendcounts[world.rank()]);
  auto local_result = std::vector<double>(sendcounts[world.rank()]);
  boost::mpi::scatterv(world, input.data(), sendcounts, local_input.data(), 0);

  const size_t radix = 8;
  const size_t histogram_size = 1 << radix;
  const size_t bits_per_item = sizeof(double) * CHAR_BIT;
  const size_t histograms_count = bits_per_item / radix;
  const size_t histogram_mask = histogram_size - 1;
  auto histograms = std::vector(histograms_count, std::vector<size_t>(histogram_size, 0));

  auto get_histogram_value = [&](size_t histogram_index, double item) -> auto & {
    const auto double_internal = std::bit_cast<uint64_t>(item);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return histograms[histogram_index][((double_internal ^ (-(double_internal >> (bits_per_item - 1)) |
                                                            (static_cast<size_t>(1) << (bits_per_item - 1)))) >>
                                        (radix * histogram_index)) &
                                       histogram_mask];
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  };

  for (const auto item : local_input) {
    for (size_t i = 0; i < histograms_count; i++) {
      get_histogram_value(i, item)++;
    }
  }

  for (auto &histogram : histograms) {
    size_t sum = 0;
    for (size_t i = 0; i < histogram_size; i++) {
      size_t count = histogram[i];
      histogram[i] = sum;
      sum += count;
    }
  }
  for (size_t i = 0; i < histograms_count; i++) {
    for (const auto item : local_input) {
      size_t &dest = get_histogram_value(i, item);
      local_result[dest] = item;
      dest++;
    }
    std::swap(local_input, local_result);
  }
  if constexpr (histograms_count % 2 == 0) {
    std::swap(local_input, local_result);
  }

  boost::mpi::gatherv(world, local_result.data(), local_result.size(), input.data(), sendcounts, 0);

  if (world.rank() == 0) {
    std::vector<int> sorted_chunk_offsets(displs);
    for (size_t i = 0; i < input_length; i++) {
      std::optional<std::pair<double, size_t>> min_value_among_chunks;
      for (size_t chunk_idx = 0; chunk_idx < sorted_chunk_offsets.size(); chunk_idx++) {
        bool chunk_is_fully_utilized =
            sorted_chunk_offsets[chunk_idx] ==
            (chunk_idx == sorted_chunk_offsets.size() - 1 ? static_cast<int>(input_length) : displs[chunk_idx + 1]);
        if (!chunk_is_fully_utilized &&
            (!min_value_among_chunks || min_value_among_chunks->first > input[sorted_chunk_offsets[chunk_idx]])) {
          min_value_among_chunks = std::pair(input[sorted_chunk_offsets[chunk_idx]], chunk_idx);
        }
      }
      if (min_value_among_chunks) {
        sorted_chunk_offsets[min_value_among_chunks->second]++;
        result[i] = min_value_among_chunks->first;
      }
    }
  }

  return true;
}

bool rams_s_radix_sort_with_simple_merge_for_doubles_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result.begin(), result.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  }
  return true;
}
