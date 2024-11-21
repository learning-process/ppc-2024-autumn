#pragma once

#include <memory>  // для std::shared_ptr
#include <string>

#include "core/task/include/task.hpp"

namespace muradov_m_count_alpha_chars_seq {

class AlphaCharCountTaskSequential : public ppc::core::Task {
 public:
  explicit AlphaCharCountTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_str_;
  int alpha_count_ = 0;
};

}  // namespace muradov_m_count_alpha_chars_seq
