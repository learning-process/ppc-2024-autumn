#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace somov_i_horizontal_scheme {

class MatrixVectorTask : public ppc::core::Task {
 public:
  explicit MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData);

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;
};

}  // namespace somov_i_horizontal_scheme