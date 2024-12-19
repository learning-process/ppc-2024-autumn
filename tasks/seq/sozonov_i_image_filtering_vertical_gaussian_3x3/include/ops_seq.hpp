#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace sozonov_i_image_filtering_vertical_gaussian_3x3_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> image, filtered_image;
  int width{}, height{};
};

}  // namespace sozonov_i_image_filtering_vertical_gaussian_3x3_seq