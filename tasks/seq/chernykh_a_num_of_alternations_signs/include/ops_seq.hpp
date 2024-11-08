#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_num_of_alternations_signs_seq {

class Task : public ppc::core::Task {
 public:
  explicit Task(std::shared_ptr<ppc::core::TaskData> task_data) : ppc::core::Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input;
  int result{};
};

}  // namespace chernykh_a_num_of_alternations_signs_seq