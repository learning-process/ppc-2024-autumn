#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3 {

class TaskSeq : public ppc::core::Task {
 public:
  explicit TaskSeq(std::shared_ptr<ppc::core::TaskData> data) : Task(std::move(data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int num_rows;
  int num_cols;
  std::vector<double> output_vector;
  std::vector<double> data_matrix;
};

}  // namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3