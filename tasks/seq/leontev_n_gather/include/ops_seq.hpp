#pragma once

#include <memory>
#include <numeric>
#include <string>

#include "core/task/include/task.hpp"

namespace leontev_n_mat_vec_seq {
template <class InOutType>
class MatVecSequential : public ppc::core::Task {
 public:
  explicit MatVecSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<InOutType> vec_;
  std::vector<InOutType> mat_;
  std::vector<InOutType> res;
};

}  // namespace leontev_n_mat_vec_seq
