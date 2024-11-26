#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_readers_and_writers_seq {

class ReadersWritersTaskSequential : public ppc::core::Task {
 public:
  explicit ReadersWritersTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_{}, res{};
};

}