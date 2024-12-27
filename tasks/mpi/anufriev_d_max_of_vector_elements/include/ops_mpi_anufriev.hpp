#pragma once

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_max_of_vector_elements_parallel {

[[nodiscard]] std::vector<int32_t> make_random_vector(int32_t size, int32_t val_min, int32_t val_max);

class VectorMaxSeq : public ppc::core::Task {
 public:
  explicit VectorMaxSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();
};

class VectorMaxPar : public ppc::core::Task {
 public:
  explicit VectorMaxPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> input_, local_input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();
  boost::mpi::communicator world;
};

}  // namespace anufriev_d_max_of_vector_elements_parallel