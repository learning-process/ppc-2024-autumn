#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace tyshkevich_a_num_of_orderly_violations_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size = 0;
  std::vector<int> input_;
  std::vector<int> res;
  std::string ops;
};

}  // namespace tyshkevich_a_num_of_orderly_violations_seq