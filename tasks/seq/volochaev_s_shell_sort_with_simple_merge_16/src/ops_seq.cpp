#include "seq/volochaev_s_shell_sort_with_simple_merge_16/include/ops_seq.hpp"

#include <functional>
#include <thread>

using namespace std::chrono_literals;

bool volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16::pre_processing() {
  internal_order_test();

  int* x = reinterpret_cast<int*>(taskData->inputs[0]);
  size_ = taskData->inputs_count[0];

  mas = new int[size_];

  std::copy(x, x + size_, mas);

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16::validation() {
  internal_order_test();
  return static_cast<int>(taskData->inputs_count[0]) > 0;
}

bool volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16::run() {
  internal_order_test();

  int n = size_;
  int gap = n / 2;

  while (gap > 0) {
    for (int i = gap; i < n; ++i) {
      int temp = mas[i];
      int j = i;
      while (j >= gap && mas[j - gap] > temp) {
        mas[j] = mas[j - gap];
        j -= gap;
      }
      mas[j] = temp;
    }
    gap /= 2;
  }

  return true;
}

bool volochaev_s_shell_sort_with_simple_merge_16_seq::Lab3_16::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(mas, mas + size_, ans);

  delete[] mas;

  return true;
}
