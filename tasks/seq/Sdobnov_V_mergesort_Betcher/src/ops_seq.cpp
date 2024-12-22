// Copyright 2024 Sdobnov Vladimir
#include "seq/Sdobnov_V_mergesort_Betcher/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void Sdobnov_V_mergesort_Betcher_seq::sortPair(int& a, int& b) {
  if (a > b) {
    std::swap(a, b);
  }
}

void Sdobnov_V_mergesort_Betcher_seq::oddEvenMergeSort(std::vector<int>& a, int l, int r) {
  if (r <= l) return;

  int m = (l + r) / 2;

  oddEvenMergeSort(a, l, m);
  oddEvenMergeSort(a, m + 1, r);

  oddEvenMerge(a, l, r);
}

void Sdobnov_V_mergesort_Betcher_seq::oddEvenMerge(std::vector<int>& a, int l, int r) {
  std::cout << "Merge l: " << l << " r: " << r << std::endl;
  for (int i = 0; i < a.size(); i++) {
    std::cout << a[i] << ' ';
  }
  std::cout << std::endl;
  int len = r - l + 1;
  if (len <= 1) return;

  int m = (l + r) / 2;
  oddEvenMerge(a, l, m);
  oddEvenMerge(a, m + 1, r);

  for (int i = l; i <= m - 1; i++) {
    sortPair(a[i], a[i + 1]);
  }

  for (int i = m; i <= r; i++) {
    sortPair(a[i], a[i + 1]);
  }

  if ((r - l + 1) % 2 != 0) {
    sortPair(a[r - 1], a[r]);
    // sortPair(a[l], a[l + 1]);
  }
}

std::vector<int> Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(int size, int lower_bound,
                                                                         int upper_bound) {
  std::vector<int> res(size);
  for (int i = 0; i < size; i++) {
    res[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return res;
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::pre_processing() {
  internal_order_test();

  size_ = taskData->inputs_count[0];
  res_.assign(size_, 0);

  auto* input = reinterpret_cast<int*>(taskData->inputs[0]);

  std::copy(input, input + size_, res_.begin());
  std::cout << "Input " << std::endl;
  for (int i = 0; i < size_; i++) {
    std::cout << res_[i] << ' ';
  }
  std::cout << std::endl;

  return true;
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::validation() {
  internal_order_test();

  return (taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 && taskData->inputs.size() == 1 &&
          taskData->outputs_count.size() == 1 && taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1);
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::run() {
  internal_order_test();
  oddEvenMergeSort(res_, 0, size_ - 1);
  std::cout << "Result " << std::endl;
  for (int i = 0; i < size_; i++) {
    std::cout << res_[i] << ' ';
  }
  std::cout << std::endl;
  return true;
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::post_processing() {
  internal_order_test();
  std::cout << "Result2 " << std::endl;
  for (int i = 0; i < size_; i++) {
    std::cout << res_[i] << ' ';
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  std::cout << std::endl;
  return true;
}
