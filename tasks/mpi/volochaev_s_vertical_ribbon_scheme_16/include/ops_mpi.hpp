#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_vertical_ribbon_scheme_16_mpi {

class Lab2_16_seq : public ppc::core::Task {
 public:
  explicit Lab2_16_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* input_A;
  int* input_B;
  int m{};
  int n{};
  std::vector<int> res;
};

class Lab2_16_mpi : public ppc::core::Task {
 public:
  explicit Lab2_16_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int m{};
  int n{};

  struct Data {
    int m;
    int n;
    std::vector<int> distribution;
    std::vector<int> displacement;
    std::vector<int> input_A1;
    std::vector<int> input_B1;
  };

  std::vector<int> res;
  std::vector<int> distribution;
  std::vector<int> displacement;
  std::vector<int> input_A1;
  std::vector<int> input_B1;
  boost::mpi::communicator world;
};

}  // namespace volochaev_s_vertical_ribbon_scheme_16_mpi
