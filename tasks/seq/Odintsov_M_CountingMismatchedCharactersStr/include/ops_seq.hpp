
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace Odintsov_M_CountingMismatchedCharactersStr_seq {

class CountingCharacterSequential : public ppc::core::Task {
 public:
  explicit CountingCharacterSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char*> input;
  int ans{};
};

}  // namespace Odintsov_M_CountingMismatchedCharactersStr_seq