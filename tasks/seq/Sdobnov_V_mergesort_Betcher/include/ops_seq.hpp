// Copyright 2024 Sdobnov Vladimir
#pragma once
#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace Sdobnov_V_mergesort_Betcher_seq {

void sortPair(int& a, int& b);
void batchersort(std::vector<int>& a, int l, int r);
std::vector<int> generate_random_vector(int size, int lower_bound = 0, int upper_bound = 50);

class MergesortBetcherSeq : public ppc::core::Task {
 public:
  explicit MergesortBetcherSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> res_;
  int size_;
};
}  // namespace Sdobnov_V_mergesort_Betcher_seq