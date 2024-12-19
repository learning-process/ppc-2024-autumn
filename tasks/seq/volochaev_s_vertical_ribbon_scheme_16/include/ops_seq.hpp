#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_vertical_ribbon_scheme_16_seq {

class Lab2_16 : public ppc::core::Task {
 public:
  explicit Lab2_16(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* input_A;
  int* input_B;
  int m{};
  int n{};
  std::vector<int> res;
};

}  // namespace volochaev_s_vertical_ribbon_scheme_16_seq
