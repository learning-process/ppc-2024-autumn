// Copyright 2024 Sedova Olga
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

namespace sedova_o_vertical_ribbon_scheme_mpi {

class ParallelMPI : public ppc::core::Task {
 public:
  explicit ParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows_{};
  int cols_{};
  std::vector<int> input_matrix_1;
  std::vector<int> input_vector_1;
  std::vector<int> result_vector_;
  std::vector<int> proc;
  std::vector<int> off;
  boost::mpi::communicator world;
};

class SequentialMPI : public ppc::core::Task {
 public:
  explicit SequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* matrix_;
  int* vector_;
  std::vector<int> result_vector_;
  int rows_;
  int cols_;
};
}  // namespace sedova_o_vertical_ribbon_scheme_mpi