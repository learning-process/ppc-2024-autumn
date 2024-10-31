#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "core/task/include/task.hpp"

using namespace std::chrono_literals;

namespace kholin_k_vector_neighbor_diff_elems_seq {

template <class TypeElem>
std::vector<TypeElem> get_random_vector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<TypeElem> vec(sz);

  if (std::is_integral<TypeElem>::value) {
    std::uniform_int_distribution<int> dist(0, 999);
    for (int i = 0; i < sz; i++) {
      vec[i] = dist(gen);
    }
  } else if (std::is_floating_point<TypeElem>::value) {
    std::uniform_real_distribution<float> dist(0.0, 999.0);
    for (int i = 0; i < sz; i++) {
      vec[i] = dist(gen);
    }
  } else {
    throw std::invalid_argument("TypeElem must be an integral or floating point type");
  }

  return vec;
}

template <class TypeElem, class TypeIndex>
class MostDiffNeighborElements : public ppc::core::Task {
 public:
  explicit MostDiffNeighborElements(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<TypeElem> input_;
  double result;
  TypeIndex left_index;
  TypeIndex right_index;
  TypeElem left_elem;
  TypeElem right_elem;
};

template <class TypeElem, class TypeIndex>
bool MostDiffNeighborElements<TypeElem, TypeIndex>::pre_processing() {
  internal_order_test();
  input_ = std::vector<TypeElem>(taskData->inputs_count[0]);
  auto ptr = reinterpret_cast<TypeElem*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  result = {};
  left_index = {};
  right_index = 2;
  left_elem = {};
  right_elem = {};
  return true;
}

template <class TypeElem, class TypeIndex>
bool MostDiffNeighborElements<TypeElem, TypeIndex>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
}

template <class TypeElem, class TypeIndex>
bool MostDiffNeighborElements<TypeElem, TypeIndex>::run() {
  internal_order_test();
  double max_delta = 0;
  double delta = 0;
  size_t curr_index = 0;
  auto iter_curr = input_.begin();
  auto iter_next = iter_curr + 1;
  auto iter_end = input_.end() - 1;
  auto iter_begin = input_.begin();
  while (iter_curr != iter_end) {
    delta = fabs((double)(*iter_next - *iter_curr));
    if (delta > max_delta) {
      if (iter_begin == iter_curr) {
        curr_index = 0;
        max_delta = delta;
      } else {
        curr_index = std::distance(input_.begin(), iter_curr);
        max_delta = delta;
      }
    }
    iter_curr++;
    iter_next = iter_curr + 1;
  }
  result = max_delta;
  right_index = curr_index + 1;
  left_index = curr_index;
  left_elem = input_[left_index];

  right_elem = input_[right_index];
  return true;
}

template <class TypeElem, class TypeIndex>
bool MostDiffNeighborElements<TypeElem, TypeIndex>::post_processing() {
  internal_order_test();
  reinterpret_cast<TypeElem*>(taskData->outputs[0])[0] = left_elem;
  reinterpret_cast<TypeElem*>(taskData->outputs[0])[1] = right_elem;
  reinterpret_cast<TypeIndex*>(taskData->outputs[1])[0] = left_index;
  reinterpret_cast<TypeIndex*>(taskData->outputs[1])[1] = right_index;
  return true;
}
}  // namespace kholin_k_vector_neighbor_diff_elems_seq