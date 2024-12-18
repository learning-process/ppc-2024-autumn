#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_count_sentences {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string text{};
  int res{};
};

}  // namespace shkurinskaya_e_count_sentences