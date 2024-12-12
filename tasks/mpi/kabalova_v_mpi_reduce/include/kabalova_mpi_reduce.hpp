// Copyright 2024 Kabalova Valeria
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <numeric>
#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace kabalova_v_mpi_reduce {
bool checkValidOperation(const std::string& ops);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int result{};
  boost::mpi::communicator world;
  std::string ops;
};

}  // namespace kabalova_v_mpi_reduce