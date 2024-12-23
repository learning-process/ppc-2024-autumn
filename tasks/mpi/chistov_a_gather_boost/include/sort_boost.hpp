#pragma once
#include "mpi/chistov_a_gather_boost/include/gather_boost.hpp"

namespace chistov_a_gather_boost {

template <typename T>
class Reference : public ppc::core::Task {
 public:
  explicit Reference(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_data;
  std::vector<T> gathered_data;
  int count{};
  boost::mpi::communicator world;
};

}  // namespace chistov_a_gather_boost
