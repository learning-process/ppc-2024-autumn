// Copyright 2024 Alputov Ivan
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_most_diff_neighb_elem_mpi {

int Max_Neighbour_Seq_Pos(const std::vector<int>& data);
std::vector<int> RandomVector(int sz);

class MPISequentialTask : public ppc::core::Task {
 public:
  explicit MPISequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int result[2];
  std::vector<int> inputData;
};

class MPIParallelTask : public ppc::core::Task {
 public:
  explicit MPIParallelTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int getElementsPerProcess() const;

 private:
  boost::mpi::communicator world;
  int result[2];
  int localMaxDiff[3];
  std::vector<int> inputData, localData;
};
}  // namespace alputov_i_most_diff_neighb_elem_mpi