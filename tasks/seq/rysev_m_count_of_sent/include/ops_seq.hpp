#pragma once

#include "core/task/include/task.hpp"
#include <string>
#include <vector>

namespace rysev_m_count_of_sent_seq {
class SentCountSequential : public ppc::core::Task {
 public:
  explicit SentCountSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
 private:
  std::string input_{};
  int count{};
};
}