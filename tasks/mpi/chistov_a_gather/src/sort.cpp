#include "mpi/chistov_a_gather/include/sort.hpp"

namespace chistov_a_gather {

template <typename T>
void merge_sorted_vectors(std::vector<T>& data, int count, int rankcount) {
  std::vector<T> merged_data;
  merged_data.reserve(data.size());

  std::vector<int> indices(rankcount, 0);

  while (true) {
    T min_value = std::numeric_limits<T>::max();
    int min_index = -1;

    for (int i = 0; i < rankcount; ++i) {
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
bool Reference<T>::pre_processing() {
  internal_order_test();

  count = taskData->inputs_count[0];
  input_data = std::vector<T>(count);
  memcpy(input_data.data(), reinterpret_cast<T*>(taskData->inputs[0]), count * sizeof(T));
  std::sort(input_data.begin(), input_data.end());

  return true;
}

template <typename T>
bool Reference<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (!taskData->inputs.empty() && taskData->outputs_count[0] == taskData->inputs_count[0] * world.size());
  }
  return true;
}

template <typename T>
bool Reference<T>::run() {
  internal_order_test();

  boost::mpi::gather(world, input_data.data(), count, gathered_data, 0);

  return true;
}

template <typename T>
bool Reference<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    merge_sorted_vectors(gathered_data, count, world.size());
    std::memcpy(reinterpret_cast<T*>(taskData->outputs[0]), gathered_data.data(), gathered_data.size() * sizeof(T));
  }
  return true;
}

template <typename T>
bool Sorting<T>::pre_processing() {
  internal_order_test();

  count = taskData->inputs_count[0];
  input_data = std::vector<T>(count);
  memcpy(input_data.data(), reinterpret_cast<T*>(taskData->inputs[0]), count * sizeof(T));
  std::sort(input_data.begin(), input_data.end());

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

  chistov_a_gather::gather<T>(world, input_data, count, gathered_data, 0);

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

template class Reference<int>;
template class Reference<double>;
template class Reference<float>;
template class Sorting<int>;
template class Sorting<double>;
template class Sorting<float>;

}  // namespace chistov_a_gather
