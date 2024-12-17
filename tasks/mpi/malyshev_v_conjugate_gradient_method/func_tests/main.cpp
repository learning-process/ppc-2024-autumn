// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

TEST(malyshev_v_conjugate_gradient_method_mpi, Test_CG_Method_1_1) {
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

  malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<double> expected_result = {0.1};
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < global_result.size(); ++i) {
      ASSERT_NEAR(global_result[i], expected_result[i], epsilon);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, Test_CG_Method_2_2) {
  boost::mpi::communicator world;

  size_t matrix_size = 2;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {4.0, 1.0, 1.0, 3.0};
  std::vector<double> Values = {1.0, 2.0};
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

  malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<double> expected_result = {0.25, 0.5};
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < global_result.size(); ++i) {
      ASSERT_NEAR(global_result[i], expected_result[i], epsilon);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, Test_CG_Method_3_3) {
  boost::mpi::communicator world;

  size_t matrix_size = 3;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0};
  std::vector<double> Values = {1.0, 2.0, 3.0};
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

  malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel taskPar(taskDataPar);
  ASSERT_TRUE(taskPar.validation());
  taskPar.pre_processing();
  taskPar.run();
  taskPar.post_processing();

  std::vector<double> expected_result = {0.2, 0.4, 0.6};
  if (world.rank() == 0) {
    for (unsigned int i = 0; i < global_result.size(); ++i) {
      ASSERT_NEAR(global_result[i], expected_result[i], epsilon);
    }
  }
}