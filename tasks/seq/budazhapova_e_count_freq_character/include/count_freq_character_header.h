#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "core/task/include/task.hpp"

namespace budazhapova_e_count_freq_character_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  char symb{};
  int res{};
};

}  // namespace budazhapova_e_count_freq_character_seq