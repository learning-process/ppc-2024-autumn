#pragma once
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace baranov_a_num_of_orderly_violations_seq {
template <class iotype, class cntype>
class num_of_orderly_violations : public ppc::core::Task {
 public:
  explicit num_of_orderly_violations(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

  cntype seq_proc(std::vector<iotype> vec);

 private:
  std::vector<iotype> input_;
  cntype num_;
};

}  // namespace baranov_a_num_of_orderly_violations_seq
