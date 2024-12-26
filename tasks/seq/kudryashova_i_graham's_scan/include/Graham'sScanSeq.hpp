#pragma once
#include <algorithm>
#include <random>

#include "core/task/include/task.hpp"
namespace kudryashova_i_graham_scan_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  std::vector<int8_t> runGrahamScan(std::vector<int8_t>& input_data);

  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int8_t> input_data;
  std::vector<int8_t> result_vec;
  std::vector<std::pair<int8_t, int8_t>> pointList;
};
}  // namespace kudryashova_i_graham_scan_seq