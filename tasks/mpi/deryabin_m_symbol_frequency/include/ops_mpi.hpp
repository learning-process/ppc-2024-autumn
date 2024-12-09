#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_symbol_frequency_mpi {

class SymbolFrequencyMPITaskSequential : public ppc::core::Task {
 public:
  explicit SymbolFrequencyMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_str_{};
  int frequency_{};
  char input_symbol_{};
};
class SymbolFrequencyMPITaskParallel : public ppc::core::Task {
 public:
  explicit SymbolFrequencyMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_str_{}, local_input_str_{};
  int frequency_{}, local_found_{};
  char input_symbol_{};
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_symbol_frequency_mpi
