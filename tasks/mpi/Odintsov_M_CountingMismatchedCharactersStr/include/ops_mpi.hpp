
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace Odintsov_M_CountingMismatchedCharactersStr_mpi {

class CountingCharacterMPISequential : public ppc::core::Task {
 public:
  explicit CountingCharacterMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char*> input;
  int ans{};
};

class CountingCharacterMPIParallel : public ppc::core::Task {
 public:
  explicit CountingCharacterMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::string> local_input;
  std::vector<char*> input;
  int ans{};
  boost::mpi::communicator com;
};
}  // namespace Odintsov_M_CountingMismatchedCharactersStr_mpi