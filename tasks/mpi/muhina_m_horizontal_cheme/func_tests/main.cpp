// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "mpi/muhina_m_horizontal_cheme/include/ops_mpi.hpp"


TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_Validation) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;  
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_cols = 0;

  if (world.rank() == 0) {
    vec = {1, 1, 1}; 
    result.resize(num_cols, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(matrixVecMultParalle.validation());
  } else {
    EXPECT_TRUE(matrixVecMultParalle.validation());
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result(4,0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    vec = {1, 2, 3};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle.validation());
  ASSERT_TRUE(matrixVecMultParalle.pre_processing());
  ASSERT_TRUE(matrixVecMultParalle.run());
  ASSERT_TRUE(matrixVecMultParalle.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_result = {14, 32, 50, 68};
    ASSERT_EQ(result.size(), expected_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected_result[i]);
    }
  }

}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_RepeatingValues) {
  boost::mpi::communicator world;

  std::vector<int> matrix(999, 1);  
  std::vector<int> vec(3, 1);        
  std::vector<int> result(333, 0);  

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);
  ASSERT_EQ(matrixVecMultParalle.validation(), true);
  matrixVecMultParalle.pre_processing();
  matrixVecMultParalle.run();
  matrixVecMultParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result(333, 3);  
    ASSERT_EQ(result, expected_result);
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_NegativeValues) {
  boost::mpi::communicator world;

  std::vector<int> matrix = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12};
  std::vector<int> vec = {-1, -1, -1};
  std::vector<int> result(4, 0);  

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);
  ASSERT_EQ(matrixVecMultParalle.validation(), true);
  matrixVecMultParalle.pre_processing();
  matrixVecMultParalle.run();
  matrixVecMultParalle.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {6, 15, 24, 33};
    ASSERT_EQ(result.size(), expected_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected_result[i]);
    }
  }
}