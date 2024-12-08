
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovyev_d_vector_max_mpi {
int vectorMax(std::vector<int, std::allocator<int>> v);
class VectorMaxSequential : public ppc::core::Task {
 public:
  explicit VectorMaxSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data;
  int result{};
  std::string ops;
};

}  // namespace solovyev_d_vector_max_mpi