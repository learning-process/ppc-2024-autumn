#pragma once
#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>


#include "core/task/include/task.hpp"

namespace laganina_e_readers_writers_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> shared_data{};
  std::vector<int> res_{};
  boost::mpi::communicator world;
};

} 