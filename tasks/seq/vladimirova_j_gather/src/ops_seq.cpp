// Copyright 2024 Nesterov Alexander
#include "seq/vladimirova_j_gather/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

std::vector<int> vladimirova_j_gather_seq::noDeadEnds(std::vector<int> way) {
  int i = 0;
  size_t j = i + 1;
  while ((i >= 0) && (j < way.size())) {
    if ((way[i] == -1) && (way[i] == way[j])) {
      do {
        i -= 1;
        j += 1;
        if ((i < 0) || (!(j < way.size()))) {
          break;
        };

        if (((way[i] * way[i] == 1) && (way[i] == (-1) * way[j])) ||
            (way[i] * way[j] == 4)) {  // if rl lr or uu dd   1-1 -11 or 22 -2-2
          way[i] = 0;
          way[j] = 0;

        } else {
          break;
        }

      } while ((i > 0) && (j < way.size()));
      i = j - 1;
    }

    j++;
    i++;
  }

  std::vector<int> ans = std::vector<int>();
  for (auto k : way)
    if (k != 0) ans.push_back(k);
  // way.erase(std::remove(way.begin(), way.end(), 0), way.end());

  return ans;
}

bool vladimirova_j_gather_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  return true;
}

bool vladimirova_j_gather_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0) return false;
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    switch (tmp_ptr[i]) {
      case 1:
      case 2:
      case -1:
        break;
      default:
        return false;
    }
  }
  return true;
}

bool vladimirova_j_gather_seq::TestTaskSequential::run() {
  internal_order_test();
  res = vladimirova_j_gather_seq::noDeadEnds(input_);
  return true;
}

bool vladimirova_j_gather_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  taskData->outputs_count[0] = res.size();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), output_data);
  // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}
