#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_count_characters_27_seq {

class Lab1_27 : public ppc::core::Task {
 public:
  explicit Lab1_27(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<char, char>> input_;
  int sz1, sz2;
  int res{};
};

}  // namespace volochaev_s_count_characters_27_seq
