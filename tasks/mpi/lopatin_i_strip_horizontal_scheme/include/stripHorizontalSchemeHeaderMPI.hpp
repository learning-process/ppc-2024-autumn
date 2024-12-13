#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/status.hpp>
#include <random>

#include "core/task/include/task.hpp"

namespace lopatin_i_strip_horizontal_scheme_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> vector_;
  std::vector<int> resultVector_;

  int sizeX, sizeY;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> vector_;
  std::vector<int> resultVector_;

  std::vector<int> localMatrix;
  std::vector<int> localVector;

  int sizeX, sizeY;

  boost::mpi::communicator world;
};

}  // namespace lopatin_i_strip_horizontal_scheme_mpi