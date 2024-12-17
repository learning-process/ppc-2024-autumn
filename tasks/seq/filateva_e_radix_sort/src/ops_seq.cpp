// Filateva Elizaveta Radix Sort

#include "seq/filateva_e_radix_sort/include/ops_seq.hpp"

bool filateva_e_radix_sort_seq::RadixSort::pre_processing() {
  internal_order_test();

  this->size = taskData->inputs_count[0];
  auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
  this->arr.assign(temp, temp + size);
  this->ans.resize(size);

  return true;
}

bool filateva_e_radix_sort_seq::RadixSort::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool filateva_e_radix_sort_seq::RadixSort::run() {
  internal_order_test();
  
  int kol = 20;
  std::vector<std::list<int>> radix_list(kol);

  int raz = 10;
  for (int i = 0; i < arr.size(); i++) {
    radix_list[arr[i] % raz + 10].push_back(arr[i]);
  }
  while (radix_list[10].size() != arr.size()) {
    raz *= 10;
    std::vector<std::list<int>> temp(kol);
    for (int i = 0; i < kol; i++) {
      for (auto p : radix_list[i]) {
        temp[p % raz / (raz / 10) + 10].push_back(p);
      }
    }
    radix_list = temp;
  }

  int i = 0;
  for (auto a: radix_list[10]) {
    ans[i] = a;
    i++;
  }

  return true;
}

bool filateva_e_radix_sort_seq::RadixSort::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(ans.begin(), ans.end(), output_data);
  return true;
}
