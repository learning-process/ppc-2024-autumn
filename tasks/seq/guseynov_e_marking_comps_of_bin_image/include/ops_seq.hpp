#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace guseynov_e_marking_comps_of_bin_image_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image_;
  std::vector<int> labeled_image;
  int rows;
  int columns;
};

}  // namespace guseynov_e_marking_comps_of_bin_image_seq