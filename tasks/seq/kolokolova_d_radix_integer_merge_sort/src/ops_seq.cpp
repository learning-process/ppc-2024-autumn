#include "seq/kolokolova_d_radix_integer_merge_sort/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void kolokolova_d_radix_integer_merge_sort_seq::counting_sort_radix(std::vector<int>& array, int exp) {
  int size_vector = int(array.size());
  std::vector<int> func_res(size_vector);
  std::vector<int> nums_of_digits(20, 0);

  for (int i = 0; i < size_vector; i++) {
    int index = (array[i] / exp) % 10;
    if (array[i] < 0) {
      index += 10;
    }
    nums_of_digits[index]++;
  }

  for (int i = 1; i < 20; i++) {
    nums_of_digits[i] += nums_of_digits[i - 1];
  }

  for (int i = size_vector - 1; i >= 0; i--) {
    int index = (array[i] / exp) % 10;
    if (array[i] < 0) {
      index += 10;
    }
    func_res[nums_of_digits[index] - 1] = array[i];
    nums_of_digits[index]--;
  }

  for (int i = 0; i < size_vector; i++) {
    array[i] = func_res[i];
  }
}

std::vector<int> kolokolova_d_radix_integer_merge_sort_seq::radix_sort(std::vector<int>& array) {
  int max_num = *max_element(array.begin(), array.end());
  int min_num = *min_element(array.begin(), array.end());

  for (int exp = 1; max_num / exp > 0 || min_num / exp < 0; exp *= 10) {
    counting_sort_radix(array, exp);
  }

  std::vector<int> sorted_array;
  std::vector<int> negatives;
  std::vector<int> positives;

  for (int num : array) {
    if (num < 0) {
      negatives.push_back(num);
    } else {
      positives.push_back(num);
    }
  }

  sort(negatives.begin(), negatives.end());
  sorted_array.insert(sorted_array.end(), negatives.begin(), negatives.end());
  sorted_array.insert(sorted_array.end(), positives.begin(), positives.end());

  return sorted_array;
}

bool kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_vector = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr_input = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_vector[i] = tmp_ptr_input[i];
  }
  res.resize(int(input_vector.size()));
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] != 0);
}

bool kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential::run() {
  internal_order_test();
  res = radix_sort(input_vector);
  return true;
}

bool kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_ptr[i] = res[i];
  }
  return true;
}
