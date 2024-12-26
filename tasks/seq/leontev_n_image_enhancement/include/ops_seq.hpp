#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_image_enhancement_seq {
class ImgEnhancementSequential : public ppc::core::Task {
 public:
  explicit ImgEnhancementSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> I;
  std::vector<int> image_input;
  std::vector<int> image_output;
};
}  // namespace leontev_n_image_enhancement_seq