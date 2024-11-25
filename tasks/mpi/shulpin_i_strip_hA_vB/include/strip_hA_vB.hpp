#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_strip_scheme_A_B {

std::vector<int> get_RND_matrix(int row, int col);
void calculate_seq(int rows_a, int cols_a, int cols_b, std::vector<int> A_seq, std::vector<int> B_seq, std::vector<int>& C_seq);
void calculate_mpi(int rows_a, int cols_a, int cols_b, std::vector<int> A_mpi, std::vector<int> B_mpi, std::vector<int>& C_mpi);
//void make_size_and_displace(int rows, int cols, int num_proc, std::vector<int>& size, std::vector<int>& displace);

  class Matrix_hA_vB_seq : public ppc::core::Task {
 public:
  explicit Matrix_hA_vB_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int seq_cols_A{};
  int seq_rows_A{};
  int seq_cols_B{};
  int seq_rows_B{};

  std::vector<int> seq_A{}; 
  std::vector<int> seq_B{}; 
  std::vector<int> seq_result{};
};

class Matrix_hA_vB_par : public ppc::core::Task {
 public:
  explicit Matrix_hA_vB_par(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int mpi_cols_A{};
  int mpi_rows_A{};
  int mpi_cols_B{};
  int mpi_rows_B{};

  std::vector<int> mpi_A{};
  std::vector<int> mpi_B{}; 
  std::vector<int> mpi_result{};

  std::vector<int> size{};
  std::vector<int> displ{};

  boost::mpi::communicator world;
};

}  // namespace shulpin_strip_scheme_A_B