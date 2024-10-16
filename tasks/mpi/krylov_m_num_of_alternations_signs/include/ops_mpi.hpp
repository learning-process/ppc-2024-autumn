#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace krylov_m_num_of_alternations_signs_mpi {

using namespace std::chrono_literals;

template <class ElementType, class CountType>
class TestMPITaskParallel : public ppc::core::Task {
  static_assert(sizeof(CountType) <=
                    sizeof(typename decltype(std::declval<ppc::core::TaskData>().inputs_count)::value_type),
                "There's no sense in providing CountType that exceeds TaskData capabilities");

  static bool distribute(std::vector<int>& distribution, std::vector<int>& displacement, int amount, int world_size) {
    const int average = amount / world_size;
    if (average < world_size) {
      distribution.resize(world_size, 0);
      distribution[0] = amount;
      displacement.resize(world_size, 0);
      return false;
    }

    distribution.resize(world_size, average);
    displacement.resize(world_size);

    const int leftover = amount % world_size;

    int pos = 0;
    for (int i = 0; i < world_size; i++) {
      if (i < leftover) {
        distribution[i]++;
      }
      displacement[i] = pos;
      pos += distribution[i];
    }

    return true;
  }

  static int calc_distribution(int world_rank, int amount, int world_size) {
    const int average = amount / world_size;
    const int leftover = amount % world_size;
    if (average < world_size && world_rank != 0) {
      return 0;
    }
    return average + ((world_rank < leftover) ? 1 : 0);
  }

 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override {
    internal_order_test();

    res = 0;

    unsigned int amount = 0;
    if (world.rank() == 0) {
      amount = taskData->inputs_count[0];
    }
    boost::mpi::broadcast(world, amount, 0);

    if (world.rank() == 0) {
      std::vector<int> distribution;
      std::vector<int> displacement;
      if (distribute(distribution, displacement, amount, world.size())) {
        std::transform(std::begin(distribution), std::end(distribution) - 1, std::begin(distribution),
                       [](auto x) { return x + 1; });
      }

      partial_input_.resize(distribution[0]);

      const auto* in_p = reinterpret_cast<ElementType*>(taskData->inputs[0]);
      boost::mpi::scatterv(world, in_p, distribution, displacement, partial_input_.data(), distribution[0], 0);
    } else {
      int distribution = calc_distribution(world.rank(), amount, world.size());
      if (distribution > 0) {
        if (world.rank() != world.size() - 1) {
          distribution++;
        }
        partial_input_.resize(distribution);
        boost::mpi::scatterv(world, partial_input_.data(), distribution, 0);
      }
    }

    return true;
  }

  bool validation() override {
    internal_order_test();

    return world.rank() != 0 || (taskData->outputs_count[0] == 1);
  }

  bool run() override {
    internal_order_test();

    CountType partial_res = 0;

    const std::size_t size = partial_input_.size();
    if (size > 0) {
      bool neg = partial_input_[0] < 0;
      for (std::size_t i = 1; i < size; i++) {
        bool cur = partial_input_[i] < 0;
        if (neg == cur) {
          continue;
        }
        partial_res++;
        neg = cur;
      }
    }

    boost::mpi::reduce(world, partial_res, res, std::plus(), 0);

    std::this_thread::sleep_for(20ms);

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    if (world.rank() == 0) {
      reinterpret_cast<CountType*>(taskData->outputs[0])[0] = res;
    }

    return true;
  }

 private:
  std::vector<ElementType> partial_input_{};
  CountType res{};
  boost::mpi::communicator world;
};

template <class ElementType, class CountType>
class TestMPITaskSequential : public ppc::core::Task {
  static_assert(sizeof(CountType) <=
                    sizeof(typename decltype(std::declval<ppc::core::TaskData>().inputs_count)::value_type),
                "There's no sense in providing CountType that exceeds TaskData capabilities");

 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

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

}  // namespace krylov_m_num_of_alternations_signs_mpi
