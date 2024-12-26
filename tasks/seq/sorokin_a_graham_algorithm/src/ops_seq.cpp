// Copyright 2024 Nesterov Alexander
#include "seq/sorokin_a_graham_algorithm/include/ops_seq.hpp"

std::vector<int> grahamAlg(const std::vector<int> &input) {
  int n = input.size() / 2;
  if (n <= 3) {
    return input;
  }
  std::vector<int> indices(n);
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  int minind = 0;
  for (int i = 0; i < n; ++i) {
    if (input[i * 2 + 1] < input[minind * 2 + 1]) {
      minind = i;
    }
    if (input[i * 2 + 1] == input[minind * 2 + 1]) {
      if (input[i * 2] < input[minind * 2]) {
        minind = i;
      }
    }
  }
  std::sort(indices.begin(), indices.end(), [&input, minind](int p1, int p2) {
    if (p1 == minind) return true;
    if (p2 == minind) return false;
    int x0 = input[minind * 2];
    int y0 = input[minind * 2 + 1];
    int x1 = input[p1 * 2];
    int y1 = input[p1 * 2 + 1];
    int x2 = input[p2 * 2];
    int y2 = input[p2 * 2 + 1];
    int orient = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (orient == 0) {
      int dx1 = input[p1 * 2] - input[minind * 2];
      int dy1 = input[p1 * 2 + 1] - input[minind * 2 + 1];
      int dx2 = input[p2 * 2] - input[minind * 2];
      int dy2 = input[p2 * 2 + 1] - input[minind * 2 + 1];
      return dx1 * dx1 + dy1 * dy1 < dx2 * dx2 + dy2 * dy2;
    }
    return orient > 0;
  });

  std::vector<int> stack;

  stack.emplace_back(indices[0]);
  stack.emplace_back(indices[1]);

  for (int i = 2; i < n; ++i) {
    while (stack.size() > 1) {
      int top = stack.back();
      int nextToTop = stack[stack.size() - 2];

      int x0 = input[nextToTop * 2];
      int y0 = input[nextToTop * 2 + 1];
      int x1 = input[top * 2];
      int y1 = input[top * 2 + 1];
      int x2 = input[indices[i] * 2];
      int y2 = input[indices[i] * 2 + 1];

      int orient = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);

      if (orient <= 0) {
        stack.pop_back();
      } else {
        break;
      }
    }
    stack.emplace_back(indices[i]);
  }

  std::vector<int> out;
  for (int index : stack) {
    out.emplace_back(input[2 * index]);
    out.emplace_back(input[2 * index + 1]);
  }
  return out;
}

bool sorokin_a_graham_algorithm_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool sorokin_a_graham_algorithm_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[0] % 2 == 0;
}

bool sorokin_a_graham_algorithm_seq::TestTaskSequential::run() {
  internal_order_test();
  res_ = grahamAlg(input_);
  return true;
}

bool sorokin_a_graham_algorithm_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto *tmp_ptr = reinterpret_cast<int *>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), tmp_ptr);
  return true;
}
