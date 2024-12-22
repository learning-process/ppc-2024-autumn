#include "seq/baranov_a_odd_even_mergesort/include/header_seq_odd_even.hpp"
namespace baranov_a_qsort_odd_even_merge_seq {

template <typename iotype>
std::vector<iotype> odd_even_mergesort_seq<iotype>::q_sort_stack(std::vector<iotype>& vec_) {
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

template <class iotype>
bool odd_even_mergesort_seq<iotype>::pre_processing() {
  internal_order_test();
  // Init vectors
  int n = taskData->inputs_count[0];
  input_ = std::vector<iotype>(n);
  void* ptr_r = taskData->inputs[0];
  void* ptr_d = input_.data();
  memcpy(ptr_d, ptr_r, sizeof(iotype) * n);

  return true;
}
template <class iotype>
bool odd_even_mergesort_seq<iotype>::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 1);
}
template <class iotype>
bool odd_even_mergesort_seq<iotype>::run() {
  internal_order_test();
  output_ = input_;
  output_ = q_sort_stack(output_);
  return true;
}
template <class iotype>
bool odd_even_mergesort_seq<iotype>::post_processing() {
  internal_order_test();

  for (int i = 0, vec_size_ = input_.size(); i != vec_size_; ++i) {
    reinterpret_cast<iotype*>(taskData->outputs[0])[i] = output_[i];
  }
  return true;
}

template class odd_even_mergesort_seq<int>;
template class odd_even_mergesort_seq<double>;
template class odd_even_mergesort_seq<unsigned>;
}  // namespace baranov_a_qsort_odd_even_merge_seq