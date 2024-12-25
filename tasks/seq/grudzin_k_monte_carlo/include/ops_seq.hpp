#pragma once

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#define functionData double(std::array<double, dimension>&)

namespace grudzin_k_montecarlo_seq {

template <const int dimension>
class MonteCarloSeq : public ppc::core::Task {
 public:
  explicit MonteCarloSeq(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<functionData> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> dim;
  std::function<functionData> f;
  double result;
  int N;
};
}  // namespace grudzin_k_montecarlo_seq