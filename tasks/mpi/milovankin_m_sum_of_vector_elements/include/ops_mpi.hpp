#pragma once

#include <boost/mpi.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_sum_of_vector_elements_parallel {
// No changes to seq version
class VectorSumSeq : public ppc::core::Task {
 public:
  explicit VectorSumSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_;
  int64_t sum_ = 0;
};

class VectorSumPar : public ppc::core::Task {
 public:
  explicit VectorSumPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_, local_input_;
  int64_t sum_ = 0;
  boost::mpi::communicator world;
};

}  // namespace milovankin_m_sum_of_vector_elements_parallel
