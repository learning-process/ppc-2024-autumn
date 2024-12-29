#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

using bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_seq {

class RadixBatcherMergesortSequential : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void sort(vector<bigint>& arr);

 private:
  vector<bigint> array;  // input unsorted numbers array
  int n = 0;             // array size

  void static radix_sort(vector<bigint>& arr, bool invert);
  static void counting_sort(vector<bigint>& arr, bigint digit_place);
  static int get_number_digit_capacity(bigint num);
};

}  // namespace belov_a_radix_batcher_mergesort_seq

#endif  // OPS_SEQ_HPP