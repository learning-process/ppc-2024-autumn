#include "seq/gordeeva_t_shell_sort_batcher_merge/include/ops_seq.hpp"

using namespace std::chrono_literals;

void gordeeva_t_shell_sort_batcher_merge_seq::shellSort(std::vector<int>& arr, int arr_length) {
  for (int step = arr_length / 2; step > 0; step /= 2) {
    for (int i = step; i < arr_length; i++) {
      int j = i;
      while (j >= step && arr[j - step] > arr[j]) {
        std::swap(arr[j], arr[j - step]);
        j -= step;
      }
    }
  }
}

bool gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int sz = taskData->inputs_count[0];
  auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.resize(sz);
  std::copy(input_tmp, input_tmp + sz, input_.begin());

  res_.resize(sz);

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential::run() {
  internal_order_test();

  shellSort(input_, input_.size());

  for (size_t i = 0; i < input_.size(); i++) {
    res_[i] = input_[i];
  }

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);

  std::copy(res_.begin(), res_.end(), output_matr);
  return true;
}
