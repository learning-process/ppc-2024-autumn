#pragma once

#include <string>

#include "core/task/include/task.hpp"

namespace deryabin_m_symbol_frequency_seq {

class SymbolFrequencyTaskSequential : public ppc::core::Task {
 public:
  explicit SymbolFrequencyTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_str_{};
  int frequency_{};
  char input_symbol_{};
};

}  // namespace deryabin_m_symbol_frequency_seq
