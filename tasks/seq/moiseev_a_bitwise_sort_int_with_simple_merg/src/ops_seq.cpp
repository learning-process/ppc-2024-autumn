#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

int getDigit(int num, int place) { return (abs(num) / place) % 10; }

void countingSortByDigit(std::vector<int>& arr, int place) {
  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  for (size_t i = 0; i < arr.size(); i++) {
    int digit = getDigit(arr[i], place);
    count[digit]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = arr.size() - 1; i >= 0; i--) {
    int digit = getDigit(arr[i], place);
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }

  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = output[i];
  }
}

void radixSort(std::vector<int>& arr) {
  std::vector<int> negativeNumbers;
  std::vector<int> nonNegativeNumbers;

  for (int num : arr) {
    if (num < 0) {
      negativeNumbers.push_back(num);
    } else {
      nonNegativeNumbers.push_back(num);
    }
  }

  if (!nonNegativeNumbers.empty()) {
    int maxNum = *max_element(nonNegativeNumbers.begin(), nonNegativeNumbers.end());
    for (int place = 1; maxNum / place > 0; place *= 10) {
      countingSortByDigit(nonNegativeNumbers, place);
    }
  }

  if (!negativeNumbers.empty()) {
    int minNum = *min_element(negativeNumbers.begin(), negativeNumbers.end());
    for (int place = 1; abs(minNum) / place > 0; place *= 10) {
      countingSortByDigit(negativeNumbers, place);
    }
    reverse(negativeNumbers.begin(), negativeNumbers.end());
  }

  arr.clear();
  arr.insert(arr.end(), negativeNumbers.begin(), negativeNumbers.end());
  arr.insert(arr.end(), nonNegativeNumbers.begin(), nonNegativeNumbers.end());
}

bool moiseev_a_radix_merge_seq::TestSEQTaskSequential::pre_processing() {
  internal_order_test();
  const auto* src = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(src, src + taskData->inputs_count[0]);
  res_.resize(input_.size());
  return true;
}

bool moiseev_a_radix_merge_seq::TestSEQTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool moiseev_a_radix_merge_seq::TestSEQTaskSequential::run() {
  internal_order_test();

  std::copy(input_.begin(), input_.end(), res_.begin());
  radixSort(res_);

  return true;
}

bool moiseev_a_radix_merge_seq::TestSEQTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
