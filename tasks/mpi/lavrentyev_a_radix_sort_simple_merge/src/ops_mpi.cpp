#include "mpi/lavrentyev_a_radix_sort_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <queue>
#include <vector>

void lavrentyev_a_radix_sort_simple_merge_mpi::mergeSortedVectorsInPlace(std::vector<double>& arr,
                                                                         const std::vector<int>& sizes,
                                                                         const std::vector<int>& displs) {
  std::vector<double> result;
  using Element = std::pair<double, std::pair<int, int>>;
  std::priority_queue<Element, std::vector<Element>, std::greater<>> min_heap;

  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 0) {
      min_heap.push({arr[displs[i]], {static_cast<int>(i), 0}});
    }
  }

  while (!min_heap.empty()) {
    auto [value, indices] = min_heap.top();
    min_heap.pop();

    result.push_back(value);

    int vec_index = indices.first;
    int elem_index = indices.second;

    if (elem_index + 1 < sizes[vec_index]) {
      min_heap.push({arr[displs[vec_index] + elem_index + 1], {vec_index, elem_index + 1}});
    }
  }

  arr = std::move(result);
}

bool lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge::validation() {
  internal_order_test();

  if (world.rank() != 0) return true;

  return taskData->inputs_count[0] > 1;
}

bool lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge::pre_processing() {
  internal_order_test();

  wsize = world.size();
  wrank = world.rank();

  vsize = taskData->inputs_count[0];

  if (world.rank() == 0) {
    auto* vec_data = reinterpret_cast<double*>(taskData->inputs[0]);

    arr.assign(vec_data, vec_data + vsize);
  }

  sizes.resize(wsize, 0);
  displs.resize(wsize, 0);

  for (int i = 0; i < wsize; i++) {
    sizes[i] = vsize / wsize + (i < vsize % wsize ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sizes[i - 1];
  }

  local_arr.resize(sizes[wrank]);

  return true;
}

bool lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge::run() {
  internal_order_test();

  boost::mpi::scatterv(world, arr.data(), sizes, displs, local_arr.data(), sizes[wrank], 0);

  radixSortDouble(local_arr.begin(), local_arr.end());

  if (wrank == 0) arr.clear();
  boost::mpi::gatherv(world, local_arr.data(), local_arr.size(), arr.data(), sizes, displs, 0);
  if (wrank == 0) mergeSortedVectorsInPlace(arr, sizes, displs);

  return true;
}

bool lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out_vector_ = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(arr.begin(), arr.end(), out_vector_);
  }

  return true;
}
