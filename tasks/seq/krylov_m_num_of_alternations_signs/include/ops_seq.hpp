#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace krylov_m_num_of_alternations_signs_seq {

using namespace std::chrono_literals;

template <class ElementType, class CountType>
class TestTaskSequential : public ppc::core::Task {
  static_assert(sizeof(CountType) <=
                    sizeof(typename decltype(std::declval<ppc::core::TaskData>().inputs_count)::value_type),
                "There's no sense in providing CountType that exceeds TaskData capabilities");

 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override {
    internal_order_test();

    const auto count = taskData->inputs_count[0];
    const auto* in_p = reinterpret_cast<ElementType*>(taskData->inputs[0]);
    input_.resize(count);
    std::copy(in_p, in_p + count, std::begin(input_));
    //
    res = 0;

    return true;
  }

  bool validation() override {
    internal_order_test();

    return taskData->outputs_count[0] == 1;
  }

  bool run() override {
    internal_order_test();

    const std::size_t size = input_.size();
    if (size > 0) {
      bool neg = input_[0] < 0;
      for (std::size_t i = 1; i < size; i++) {
        bool cur = input_[i] < 0;
        if (neg == cur) {
          continue;
        }
        res++;
        neg = cur;
      }
    }

    std::this_thread::sleep_for(20ms);

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    reinterpret_cast<CountType*>(taskData->outputs[0])[0] = res;

    return true;
  }

 private:
  std::vector<ElementType> input_{};
  CountType res{};
};

}  // namespace krylov_m_num_of_alternations_signs_seq
