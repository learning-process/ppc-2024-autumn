// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_contrast_adjustment_seq {

struct Pixel {
  uint8_t red;
  uint8_t green;
  uint8_t blue;

  Pixel& operator=(const uint8_t& value) {
    this->red = value;
    this->green = value;
    this->blue = value;
    return *this;
  }
};

int clamp(int value, int min, int max);
double getContrast(std::vector<kondratev_ya_contrast_adjustment_seq::Pixel>& array);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input_, res_;
  double contrast_;
};

}  // namespace kondratev_ya_contrast_adjustment_seq