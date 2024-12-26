#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_sobel_edges {

class SobelEdgeDetection : public ppc::core::Task {
 public:
  explicit SobelEdgeDetection(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<unsigned char> input_image_;
  std::vector<unsigned char> output_image_;
  int height_, width_;
};
}  // namespace fomin_v_sobel_edges