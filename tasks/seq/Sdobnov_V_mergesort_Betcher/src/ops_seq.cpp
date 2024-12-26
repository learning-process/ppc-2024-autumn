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

void Sdobnov_V_mergesort_Betcher_seq::batchersort(std::vector<int>& a, int l, int r) {
  int N = r - l + 1;
  for (int p = 1; p < N; p += p)
    for (int k = p; k > 0; k /= 2)
      for (int j = k % p; j + k < N; j += (k + k))
        for (int i = 0; i < N - j - k; i++)
          if ((j + i) / (p + p) == (j + i + k) / (p + p)) sortPair(a[l + j + i], a[l + j + i + k]);
}

std::vector<int> Sdobnov_V_mergesort_Betcher_seq::generate_random_vector(int size, int lower_bound, int upper_bound) {
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
  for (int i = 0; i < size_; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }

  return true;
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::validation() {
  internal_order_test();

  return (taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 && taskData->inputs.size() == 1 &&
          taskData->outputs_count.size() == 1 && taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1 &&
          taskData->inputs_count[0] > 0 && (taskData->inputs_count[0] & (taskData->inputs_count[0] - 1)) == 0);
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::run() {
  internal_order_test();
  batchersort(res_, 0, size_ - 1);
  return true;
}

bool Sdobnov_V_mergesort_Betcher_seq::MergesortBetcherSeq::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}
