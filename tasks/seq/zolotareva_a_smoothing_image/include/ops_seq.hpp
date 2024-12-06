// Copyright 2023 Nesterov Alexander
#pragma once
#include <vector>

#include "core/task/include/task.hpp"

namespace zolotareva_a_smoothing_image_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<float> create_gaussian_kernel(int radius, float sigma);
  static void convolve_rows(const std::vector<uint8_t>& input, int height, int width, const std::vector<float>& kernel,
                            std::vector<float>& temp);
  static void convolve_columns(const std::vector<float>& temp, int height, int width, const std::vector<float>& kernel,
                               std::vector<uint8_t>& output);

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> result_;
  int width_{0};
  int height_{0};
};

}  // namespace zolotareva_a_smoothing_image_seq