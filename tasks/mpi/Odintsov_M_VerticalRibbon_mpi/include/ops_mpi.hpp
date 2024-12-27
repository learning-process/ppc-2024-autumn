
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Odintsov_M_VerticalRibbon_mpi {

class VerticalRibbonMPISequential : public ppc::core::Task {
 public:
  explicit VerticalRibbonMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std ::vector<double> matrixA;
  std::vector<double> vectorB;
  std::vector<double> vectorC;
  // [0] - size, [1] - row, [2] - col
  int colA, rowA;
  int szA;
  int szB;
};

class VerticalRibbonMPIParallel : public ppc::core::Task {
 public:
  explicit VerticalRibbonMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int ribbon_sz;
  std::vector<double> local_ribbon;
  std ::vector<double> matrixA;
  std::vector<double> vectorB;
  std::vector<double> vectorC;
  std::vector<double> localC;
  // [0] - size, [1] - row, [2] - col
  int colA, rowA = 0;
  int szA;
  int szB;
  boost::mpi::communicator com;
};
}  // namespace Odintsov_M_VerticalRibbon_mpi