#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"
#include <string>
#include <vector>
#include <cstring>

using namespace std::chrono_literals;

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input = std::vector<double>(input_ptr, input_ptr + taskData->inputs_count[0]);
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return true;
  }
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  SortDouble(input);
  sort = input;
  return true;
}

bool komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  if (!sort.empty()) {
    memcpy(taskData->outputs[0], sort.data(), sort.size() * sizeof(double));
  }
  return true;
}

 void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::CountingSort(double* inp, double* out, int byteNum,
                                                                            int size) {
  auto mas = reinterpret_cast<unsigned char*>(inp);
  int counter[256] = {0};

  for (int i = 0; i < size; i++) {
    counter[mas[8 * i + byteNum]]++;
  }

  int tem = 0;
  for (int j = 0; j < 256; j++) {
    int b = counter[j];
    counter[j] = tem;
    tem += b;
  }

  for (int i = 0; i < size; i++) {
    out[counter[mas[8 * i + byteNum]]] = inp[i];
    counter[mas[8 * i + byteNum]]++;
  }
}

void komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::SortDouble(std::vector<double>& data) {
  int size = data.size();
  if (size == 0) return;

  std::vector<double> out_positives;
  out_positives.reserve(size);
  std::vector<double> out_negatives;
  out_negatives.reserve(size);

  for (double num : data) {
    (num < 0 ? out_negatives : out_positives).push_back(num);
  }

  for (double& num : out_negatives) {
    num = -num;
  }

  std::vector<double> sorted_positives(out_positives.size());
  std::vector<double> sorted_negatives(out_negatives.size());

  double* inp_ptr;
  double* out_ptr;

  if (!out_positives.empty()) {
    inp_ptr = out_positives.data();
    out_ptr = sorted_positives.data();

    for (int i = 0; i < 8; i++) {
      CountingSort(inp_ptr, out_ptr, i, out_positives.size());
      std::swap(inp_ptr, out_ptr);
    }

    if (inp_ptr != out_positives.data()) {
      std::copy(inp_ptr, inp_ptr + out_positives.size(), out_positives.begin());
    }
  }

  if (!out_negatives.empty()) {
    inp_ptr = out_negatives.data();
    out_ptr = sorted_negatives.data();

    for (int i = 0; i < 8; i++) {
      CountingSort(inp_ptr, out_ptr, i, out_negatives.size());
      std::swap(inp_ptr, out_ptr);
    }

    if (inp_ptr != out_negatives.data()) {
      std::copy(inp_ptr, inp_ptr + out_negatives.size(), out_negatives.begin());
    }
  }

  for (double& num : out_negatives) {
    num = -num;
  }

  data.clear();
  data.insert(data.end(), out_negatives.rbegin(), out_negatives.rend());
  data.insert(data.end(), out_positives.begin(), out_positives.end());
}