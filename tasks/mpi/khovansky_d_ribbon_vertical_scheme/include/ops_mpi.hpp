// Copyright 2024 Khovansky Dmitry
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

namespace khovansky_d_ribbon_vertical_scheme_mpi {

class RibbonVerticalSchemeSeq : public ppc::core::Task {
 public:
  explicit RibbonVerticalSchemeSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int* hello_matrix;
  int* hello_vector;
  int rows_count{};
  int columns_count{};
  std::vector<int> goodbye_vector;
};

class RibbonVerticalSchemeMPI : public ppc::core::Task {
 public:
  explicit RibbonVerticalSchemeMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> hello_matrix;
  std::vector<int> hello_vector;
  int rows_count{};
  int columns_count{};
  std::vector<int> rows_per_process;
  std::vector<int> rows_offsets;
  std::vector<int> goodbye_vector;
  boost::mpi::communicator world;
};

}  // namespace khovansky_d_ribbon_vector_scheme_mpi
