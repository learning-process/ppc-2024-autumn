#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_contrast {

class ContrastTaskSequential : public ppc::core::Task {
 public:
  explicit ContrastTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
};

}  // namespace shuravina_o_contrast