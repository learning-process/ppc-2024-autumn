#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_min_of_vector_elements_seq {

class MinOfVectorElementTaskSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorElementTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};

}  // namespace komshina_d_min_of_vector_elements_seq