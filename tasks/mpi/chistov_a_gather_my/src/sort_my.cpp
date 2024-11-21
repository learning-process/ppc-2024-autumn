#include "mpi/chistov_a_gather_my/include/sort_my.hpp"

namespace chistov_a_gather_my {

template <typename T>
void merge_sorted_vectors(std::vector<T>& data, int count, int rank_count) {
  std::vector<T> merged_data;
  merged_data.reserve(data.size());

  std::vector<int> indices(rank_count, 0);

  while (true) {
    T min_value = std::numeric_limits<T>::max();
    int min_index = -1;

    for (int i = 0; i < rank_count; ++i) {
      if (indices[i] < count) {
        auto current_value = data[i * count + indices[i]];
        if (current_value < min_value) {
          min_value = current_value;
          min_index = i;
        }
      }
    }

    if (min_index == -1) {
      break;
    }

    merged_data.push_back(min_value);
    indices[min_index]++;
  }

  data = std::move(merged_data);
}

template <typename T>
bool Sorting<T>::pre_processing() {
  internal_order_test();

  count = taskData->inputs_count[0];
  input_data = std::vector<T>(count);
  std::memcpy(input_data.data(), reinterpret_cast<T*>(taskData->inputs[0]), count * sizeof(T));

  return true;
}

template <typename T>
bool Sorting<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (!taskData->inputs.empty() && taskData->outputs_count[0] == taskData->inputs_count[0] * world.size());
  }
  return true;
}

template <typename T>
bool Sorting<T>::run() {
  internal_order_test();

  std::sort(input_data.begin(), input_data.end());
  chistov_a_gather_my::gather<T>(world, input_data, count, gathered_data, 0);

  return true;
}

template <typename T>
bool Sorting<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    merge_sorted_vectors(gathered_data, count, world.size());
    std::memcpy(reinterpret_cast<T*>(taskData->outputs[0]), gathered_data.data(), gathered_data.size() * sizeof(T));
  }
  return true;
}

template class Sorting<int>;
template class Sorting<double>;
template class Sorting<float>;
template class Sorting<char>;

}  // namespace chistov_a_gather_my