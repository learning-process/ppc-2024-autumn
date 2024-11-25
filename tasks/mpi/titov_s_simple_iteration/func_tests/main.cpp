// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/titov_s_simple_iteration/include/ops_mpi.hpp"

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Not_A_Diagonally_Dominate) {
  boost::mpi::communicator world;
  size_t matrix_size = 3;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {1.0, 2.0, 3.0, 14.0, 2.8, 35.0, 1.0, 6.0, 0.1};
  std::vector<double> Values = {3.0, 5.0, 4.0};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(taskPar.validation());
  }
}

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Parallel_1_1) {
  boost::mpi::communicator world;

  size_t matrix_size = 1;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {10.0};
  std::vector<double> Values = {1.0};
  double epsilon = 0.001;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  titov_s_simple_iteration_mpi::MPISimpleIterationParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  ASSERT_TRUE(taskPar.pre_processing());

}
