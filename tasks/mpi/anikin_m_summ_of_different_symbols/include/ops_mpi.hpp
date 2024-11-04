// Copyright 2024 Anikin Maksim
#pragma once

#include <boost/mpi.hpp>
#include <string>

#include "core/task/include/task.hpp"

namespace anikin_m_summ_of_different_symbols_mpi {

class SumDifSymMPISequential : public ppc::core::Task {
 public:
  explicit SumDifSymMPISequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_a, input_b;
  int result_{};
};

class SumDifSymMPIParallel : public ppc::core::Task {
 public:
  explicit SumDifSymMPIParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_a, input_b;
  std::string local_input_a, local_input_b;
  int result_{};

  boost::mpi::communicator world;
};

}  // namespace anikin_m_summ_of_different_symbols_mpi