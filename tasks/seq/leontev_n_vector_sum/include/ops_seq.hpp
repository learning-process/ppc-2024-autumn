#pragma once

#include <memory>
#include <numeric>
#include <string>

#include "core/task/include/task.hpp"

namespace leontev_n_vector_sum_seq {
template <class InOutType>
class VecSumSequential : public ppc::core::Task {
 public:
  explicit VecSumSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<InOutType> input_;
  InOutType res{};
};

}  // namespace leontev_n_vector_sum_seq
