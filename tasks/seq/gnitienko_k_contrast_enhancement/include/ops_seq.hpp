#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace gnitienko_k_contrast_enhancement_seq {

class ContrastEnhanceSeq : public ppc::core::Task {
 public:
  explicit ContrastEnhanceSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image;
  std::vector<int> res;
  double contrast_factor{};
};

}  // namespace gnitienko_k_contrast_enhancement_seq