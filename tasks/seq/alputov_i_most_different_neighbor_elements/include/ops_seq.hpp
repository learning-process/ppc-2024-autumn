#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_most_different_neighbor_elements_seq {

class most_different_neighbor_elements_seq : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<int, int>> input_;
  std::pair<int, int> res{};
};

}  // namespace alputov_i_most_different_neighbor_elements_seq