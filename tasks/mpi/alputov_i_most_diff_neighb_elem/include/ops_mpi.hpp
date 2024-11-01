// Copyright 2024 Alputov Ivan
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_most_diff_neighb_elem_mpi {

int Max_Neighbour_Seq_Pos(const std::vector<int>& data);

class MPISequentialTask : public ppc::core::Task {
 public:
  explicit MPISequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData;
  int result[2];
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
  std::vector<int> inputData, localData;
  int result[2];
  boost::mpi::communicator world;
  int localMaxDiff[3];
};

}  // namespace alputov_i_most_diff_neighb_elem_mpi