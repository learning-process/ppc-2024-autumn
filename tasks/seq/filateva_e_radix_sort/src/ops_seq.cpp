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
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool filateva_e_radix_sort_seq::RadixSort::run() {
  internal_order_test();

  int kol = 10;
  std::vector<std::list<int>> radix_list(kol);
  std::vector<std::list<int>> negativ_radix_list(kol);

  int raz = 10;
  int max_r = -1;
  for (unsigned long i = 0; i < arr.size(); i++) {
    max_r = std::max(max_r, std::abs(arr[i]));
    if (arr[i] >= 0) {
      radix_list[arr[i] % raz].push_back(arr[i]);
    } else {
      negativ_radix_list[std::abs(arr[i]) % raz].push_back(std::abs(arr[i]));
    }
  }

  while (max_r / (raz / 10) > 0) {
    raz *= 10;
    std::vector<std::list<int>> temp(kol);
    std::vector<std::list<int>> negativ_temp(kol);
    for (int i = 0; i < kol; i++) {
      for (auto p : radix_list[i]) {
        temp[p % raz / (raz / 10)].push_back(p);
      }
      for (auto p : negativ_radix_list[i]) {
        negativ_temp[p % raz / (raz / 10)].push_back(p);
      }
    }
    radix_list = temp;
    negativ_radix_list = negativ_temp;
  }

  auto rit = negativ_radix_list[0].rbegin();
  int i = 0;
  for (; rit != negativ_radix_list[0].rend(); rit++, i++) {
    ans[i] = -(*rit);
  }
  auto it = radix_list[0].begin();
  for (; it != radix_list[0].end(); it++, i++) {
    ans[i] = *it;
  }

  return true;
}

bool filateva_e_radix_sort_seq::RadixSort::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(ans.begin(), ans.end(), output_data);
  return true;
}
