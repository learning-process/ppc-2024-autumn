#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace beskhmelnova_k_jarvis_march_seq {

template <typename DataType>
DataType crossProduct(const std::vector<DataType>& p1, const std::vector<DataType>& p2,
                      const std::vector<DataType>& p3);

template <typename DataType>
bool isLeftAngle(const std::vector<DataType>& p1, const std::vector<DataType>& p2, const std::vector<DataType>& p3);

template <typename DataType>
void jarvisMarch(int& num_points, std::vector<std::vector<DataType>>& input, std::vector<DataType>& res_x,
                 std::vector<DataType>& res_y);

template <typename DataType>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int num_points{};
  std::vector<std::vector<DataType>> input;
  std::vector<DataType> res_x;
  std::vector<DataType> res_y;
};

}  // namespace beskhmelnova_k_jarvis_march_seq