// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <random>
#include <vector>

#include "mpi/muhina_m_horizontal_cheme/include/ops_mpi.hpp"

using namespace muhina_m_horizontal_cheme_mpi;

TEST(muhina_m_horizontal_cheme, Test_EmptyMatrix) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec = {1, 1, 1};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);

  if (world.rank() == 0) {
    EXPECT_FALSE(matrixVecMultParalle.validation());
  } else {
    EXPECT_TRUE(matrixVecMultParalle.validation());
  }
}
TEST(muhina_m_horizontal_cheme, Test_EmptyVector) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    result.resize(num_res, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(matrixVecMultParalle.validation());
  } else {
    EXPECT_TRUE(matrixVecMultParalle.validation());
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_Validation) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    vec = {1, 1, 1};
    result.resize(num_res, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  HorizontalSchemeMPIParallel matrixVecMultParalle(taskDataPar);
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
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    vec = {1, 2, 4};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result{17, 38, 59, 80, 101};
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    ASSERT_EQ(result.size(), expected_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected_result[i]);
    }
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_RepeatingValues) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    vec = {1, 1, 1};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result{3, 3, 3, 3, 3};
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    ASSERT_EQ(result.size(), expected_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected_result[i]);
    }
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_NegativeValues) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15};
    vec = {-1, -2, -4};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result{17, 38, 59, 80, 101};
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    ASSERT_EQ(result.size(), expected_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected_result[i]);
    }
  }
}
TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_2) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    vec = {1, 2, 4};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    std::vector<int> seq_result(result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataSeq->inputs_count.emplace_back(vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<HorizontalSchemeMPISequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(result.size(), seq_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], seq_result[i]);
    }
  }
}

TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_3) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int num_res = 5;

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    vec = {1, 2, 4, 5};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    std::vector<int> seq_result(result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataSeq->inputs_count.emplace_back(vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<HorizontalSchemeMPISequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(result.size(), seq_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], seq_result[i]);
    }
  }
}
TEST(muhina_m_horizontal_cheme, Test_MatrixVectorMultiplication_RandomMatrix) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vec;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows = 30;
  int num_cols = 30;
  int num_res = 30;

  if (world.rank() == 0) {
    matrix.resize(num_rows * num_cols);
    for (int j = 0; j < num_rows; ++j) {
      for (int i = 0; i < num_cols; ++i) {
        matrix[j * num_cols + i] = rand() % 100;
      }
    }
    vec.resize(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      vec[i] = rand() % 100;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataPar->inputs_count.emplace_back(num_res);
    taskDataPar->outputs_count.emplace_back(vec.size());
  }

  auto matrixVecMultParalle = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);

  ASSERT_TRUE(matrixVecMultParalle->validation());
  matrixVecMultParalle->pre_processing();
  matrixVecMultParalle->run();
  matrixVecMultParalle->post_processing();

  if (world.rank() == 0) {
    auto* temp = reinterpret_cast<int*>(taskDataPar->outputs[0]);
    result.insert(result.end(), temp, temp + num_res);

    std::vector<int> seq_result(result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataSeq->inputs_count.emplace_back(vec.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<HorizontalSchemeMPISequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(result.size(), seq_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], seq_result[i]);
    }
  }
}
