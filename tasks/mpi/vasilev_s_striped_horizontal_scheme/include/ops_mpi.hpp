#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_striped_horizontal_scheme_mpi {

void calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);

class StripedHorizontalSchemeParallelMPI : public ppc::core::Task {
 public:
  explicit StripedHorizontalSchemeParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int num_rows_;
  int num_cols_;
  std::vector<int> distribution;
  std::vector<int> displacement;
  boost::mpi::communicator world;
};

class StripedHorizontalSchemeSequentialMPI : public ppc::core::Task {
 public:
  explicit StripedHorizontalSchemeSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_matrix_;
  std::vector<int> input_vector_;
  std::vector<int> result_vector_;
  int num_rows_;
  int num_cols_;
};

}  // namespace vasilev_s_striped_horizontal_scheme_mpi
