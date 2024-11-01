
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kozlova_e_lexic_order {

class StringComparator : public ppc::core::Task {
 public:
  explicit StringComparator(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string str1{};
  std::string str2{};
  std::vector<int> res{};
};

}// namespace kozlova_e_lexic_order