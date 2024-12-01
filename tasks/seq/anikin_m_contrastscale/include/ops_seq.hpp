// Copyright 2024 Anikin Maksim
#pragma once

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_contrastscale_seq {

struct RGB {
  uint8_t R;
  uint8_t G;
  uint8_t B;
};

RGB getrandomRGB();
double getcontrast(std::vector<RGB>& in);

class ContrastScaleSeq : public ppc::core::Task {
 public:
  explicit ContrastScaleSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  float correction;
  std::vector<RGB> input_, output_;
};

}  // namespace anikin_m_contrastscale_seq
