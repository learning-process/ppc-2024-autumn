#pragma once
#include "mpi/chistov_a_gather/include/gather.hpp"

namespace chistov_a_gather {

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

template <typename T>
class Sorting : public ppc::core::Task {
 public:
  explicit Sorting(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

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

}  // namespace chistov_a_gather
