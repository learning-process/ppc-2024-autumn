#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_sum_of_vector_elements_seq {

class VectorSumSeq : public ppc::core::Task {
 public:
  explicit VectorSumSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_;
  int64_t sum_ = 0;
};

}  // namespace milovankin_m_sum_of_vector_elements_seq
