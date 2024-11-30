#pragma once

#include <random>

#include "core/task/include/task.hpp"

namespace lopatin_i_strip_horizontal_scheme_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> vector_;
  std::vector<int> resultVector_;

  int sizeX, sizeY;
};

}  // namespace lopatin_i_strip_horizontal_scheme_seq