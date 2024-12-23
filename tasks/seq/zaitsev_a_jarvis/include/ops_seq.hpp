#pragma once

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zaitsev_a_jarvis/include/point.hpp"

namespace zaitsev_a_jarvis_seq {

template <typename T>
class Jarvis : public ppc::core::Task {
 public:
  explicit Jarvis(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0];

    return true;
  };

  bool pre_processing() override {
    internal_order_test();

    length = taskData->inputs_count[0];

    auto* tmp_ptr = reinterpret_cast<Point<T>*>(taskData->inputs[0]);
    set.assign(tmp_ptr, tmp_ptr + length);
    convex_hull.clear();

    return true;
  };

  bool run() override {
    internal_order_test();

    if (taskData->inputs_count[0] < 3) {
      convex_hull = set;
      return true;
    }

    unsigned int start = 0;
    for (unsigned int i = 1; i < length; i++)
      if (set[i].y < set[start].y || (set[i].y == set[start].y && set[i].x < set[start].x)) start = i;

    unsigned int prev = start;
    unsigned int next;

    do {
      convex_hull.push_back(set[prev]);
      next = (prev + 1) % length;

      for (unsigned int supposed = 0; supposed < length; supposed++) {
        if (orientation(set[prev], set[supposed], set[next]) == 2) next = supposed;
        if (prev != supposed && set[next].between(set[prev], set[supposed])) next = supposed;
      }

      prev = next;

    } while (prev != start);

    return true;
  };

  bool post_processing() override {
    internal_order_test();

    auto* result_ptr = reinterpret_cast<zaitsev_a_jarvis_seq::Point<T>*>(taskData->outputs[0]);
    std::copy(convex_hull.begin(), convex_hull.end(), result_ptr);

    taskData->outputs_count[0] = convex_hull.size();

    return true;
  };

 private:
  unsigned int length;
  std::vector<zaitsev_a_jarvis_seq::Point<T>> set;
  std::vector<zaitsev_a_jarvis_seq::Point<T>> convex_hull;
};

}  // namespace zaitsev_a_jarvis_seq