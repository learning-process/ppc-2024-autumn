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

inline std::vector<int> generate_random_vector(int cs_temp, int rs_temp) {
  std::vector<int> temp(cs_temp * rs_temp);
  for (int i = 0; i < rs_temp; i++) {
    for (int j = 0; j < cs_temp; j++) {
      if (i == 0) {
        temp[i * rs_temp + j] = 0;
      } else {
        temp[i * rs_temp + j] = rand() % 1000;
      }
    }
  }
  return temp;
}

namespace dormidontov_e_min_value_by_columns_mpi {
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cs{};
  int rs{};
  std::vector<std::vector<int>> input_;
  std::vector<int> res_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cs{};
  int rs{};
  std::vector<int> input_;
  std::vector<int> minput_;
  std::vector<int> res_{};
  boost::mpi::communicator world;
};
}  // namespace dormidontov_e_min_value_by_columns_mpi