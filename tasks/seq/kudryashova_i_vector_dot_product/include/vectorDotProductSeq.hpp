#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace kudryashova_i_vector_dot_product {
std::vector<int> GetRandomVector(int size);
int vectorDotProduct(const std::vector<int>& vector1, const std::vector<int>& vector2);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_{};
  int result{};
};
}  // namespace kudryashova_i_vector_dot_product
