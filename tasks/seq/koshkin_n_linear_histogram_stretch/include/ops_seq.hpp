#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_n_linear_histogram_stretch_seq {
std::vector<int> getRandomImage(int sz);
void linearHistogramStretch(const std::vector<int>& input, std::vector<int>& output);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> I;  // ћассив ¤ркостей
  std::vector<int> image_input;
  std::vector<int> image_output;
};
}  // namespace koshkin_n_linear_histogram_stretch_seq