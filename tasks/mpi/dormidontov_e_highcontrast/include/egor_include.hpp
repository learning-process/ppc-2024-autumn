#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_highcontrast_mpi {
class ContrastS : public ppc::core::Task {
 public:
  explicit ContrastS(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  int ymin;
  int ymax;
  std::vector<int> y;
  std::vector<int> res_;
};

class ContrastP : public ppc::core::Task {
 public:
  explicit ContrastP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  int ymin;
  int ymax;
  std::vector<int> Y;
  std::vector<int> y;
  std::vector<int> res_;
  boost::mpi::communicator world;
};

}  // namespace dormidontov_e_highcontrast_mpi