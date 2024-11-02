// seq/include/ops_seq_anufriev.hpp
#pragma once

#include <limits>  // For numeric_limits
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_max_of_vector_elements_seq {

class VectorMaxSeq : public ppc::core::Task {
 public:
  explicit VectorMaxSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();  // Initialize with the smallest possible value
};

}  // namespace anufriev_d_max_of_vector_elements_seq