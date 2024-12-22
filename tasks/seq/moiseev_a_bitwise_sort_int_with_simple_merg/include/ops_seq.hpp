#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace moiseev_a_radix_merge_seq {

class TestSEQTaskSequential : public ppc::core::Task {
 public:
  explicit TestSEQTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, res_;
};

}  // namespace moiseev_a_radix_merge_seq