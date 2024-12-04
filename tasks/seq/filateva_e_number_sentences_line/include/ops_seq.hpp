// Filateva Elizaveta Number_of_sentences_per_line

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_number_sentences_line_seq {

class NumberSentencesLine : public ppc::core::Task {
 public:
  explicit NumberSentencesLine(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string line;
  int sentence_count;
};

}  // namespace filateva_e_number_sentences_line_seq