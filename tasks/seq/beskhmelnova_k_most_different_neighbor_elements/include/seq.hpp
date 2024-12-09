#pragma once

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace beskhmelnova_k_most_different_neighbor_elements_seq {

template <typename DataType>
std::vector<DataType> getRandomVector(int sz);

template <typename DataType>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  int position_of_first_neighbour_seq(std::vector<DataType> vector);

 private:
  std::vector<DataType> input_;
  DataType res[2];
};

}  // namespace beskhmelnova_k_most_different_neighbor_elements_seq