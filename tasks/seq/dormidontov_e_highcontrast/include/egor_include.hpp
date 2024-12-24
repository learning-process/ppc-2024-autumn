#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_highcontrast_seq {
class ContrastS : public ppc::core::Task {
 public:
  explicit ContrastS(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  int ymin;
  int ymax;
  std::vector<int> y;
  std::vector<int> res_;
};
}  // namespace dormidontov_e_highcontrast_seq