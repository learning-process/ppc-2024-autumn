#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zaitsev_a_jarvis/include/point.hpp"

namespace mine_seq = zaitsev_a_jarvis_seq;

namespace zaitsev_a_jarvis_mpi {

template <typename T>
class Jarvis : public ppc::core::Task {
 public:
  explicit Jarvis(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return world.rank() != root ||
           (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == taskData->inputs_count[0]);
  };

  bool pre_processing() override {
    internal_order_test();

    if (world.rank() != root) return true;

    length = taskData->inputs_count[0];

    auto* tmp_ptr = reinterpret_cast<mine_seq::Point<T>*>(taskData->inputs[0]);
    set.assign(tmp_ptr, tmp_ptr + length);

    convex_hull.clear();

    return true;
  };

  bool run() override {
    internal_order_test();

    boost::mpi::broadcast(world, set, root);

    if (set.size() < 3) {
      if (world.rank() == root) convex_hull = set;
      return true;
    }

    boost::mpi::broadcast(world, length, root);

    shift = (world.rank() == 0) ? length / world.size() + length % world.size() : length / world.size();

    unsigned int st_pos = world.rank() > root ? world.rank() * shift + length % world.size() : world.rank() * shift;
    unsigned int end_pos = st_pos + shift;

    unsigned int prev;
    unsigned int next;
    unsigned int start;

    if (world.rank() == root) {
      start = 0;
      for (unsigned int i = 1; i < length; i++)
        if (set[i].y < set[start].y || (set[i].y == set[start].y && set[i].x < set[start].x)) start = i;
    }
    boost::mpi::broadcast(world, start, root);
    prev = start;
    do {
      if (world.rank() == root) convex_hull.push_back(set[prev]);
      next = (prev + 1) % length;

      for (unsigned int supposed = st_pos; supposed < end_pos; supposed++) {
        if (orientation(set[prev], set[supposed], set[next]) == 2) next = supposed;
        if (prev != supposed && set[next].between(set[prev], set[supposed])) next = supposed;
      }

      std::vector<T> most_counterclock_wise(world.size(), 0);

      boost::mpi::gather<T>(world, next, most_counterclock_wise.data(), root);

      if (world.rank() == root) {
        for (auto supposed : most_counterclock_wise) {
          if (orientation(set[prev], set[supposed], set[next]) == 2) next = supposed;
          if ((int)prev != supposed && set[next].between(set[prev], set[supposed])) next = supposed;
        }
        prev = next;
      }
      boost::mpi::broadcast(world, prev, root);
    } while (prev != start);
    return true;
  };

  bool post_processing() override {
    internal_order_test();

    if (world.rank() != root) return true;

    auto* result_ptr = reinterpret_cast<zaitsev_a_jarvis_seq::Point<T>*>(taskData->outputs[0]);
    taskData->outputs_count[0] = convex_hull.size();
    std::copy(convex_hull.begin(), convex_hull.end(), result_ptr);
    return true;
  };

 private:
  int root = 0;
  unsigned int length;
  unsigned int shift;
  std::vector<mine_seq::Point<T>> set;
  std::vector<mine_seq::Point<T>> convex_hull;
  const boost::mpi::communicator world;
};

}  // namespace zaitsev_a_jarvis_mpi