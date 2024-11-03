// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"
std::vector<std::vector<int>> getRandomMatrix_(int n, int m) {
  int left = 0;
  int right = 10005;

  // Создаем матрицу
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

  // Заполняем матрицу случайными значениями
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = left + std::rand() % (right - left + 1);
    }
  }
  for (int i = 0; i < n; ++i) {
    int m_ = std::rand() % m;
    matrix[i][m_] = -1;
  }
  return matrix;
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isFalse0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  if (world.rank() == 0) {
    morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataSeq);
    morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isFalse1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  if (world.rank() == 0) {
    for (size_t i = 0; i < matrix.size(); ++i)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataSeq);
    morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isFalse2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  if (world.rank() == 0) {
    for (size_t i = 0; i < matrix.size(); ++i) taskDataSeq->inputs_count.emplace_back(1);
    morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataSeq);
    morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Validation_isTrue) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  std::vector<int> res = {1, 2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < matrix.size(); ++i)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(matrix[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
    ASSERT_TRUE(testMpiTaskParallel.validation());
    ASSERT_TRUE(testMpiTaskSequential.validation());
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Main0) {
  std::vector<std::vector<int>> matrixPar;
  std::vector<std::vector<int>> matrixSeq;
  const int n = 2;
  const int m = 2;
  std::vector<int32_t> resPar(n);
  std::vector<int32_t> resSeq(n);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  matrixSeq = {{1, 2}, {3, 4}};
  for (size_t i = 0; i < matrixSeq.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixSeq[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  if (world.rank() == 0) {
    matrixPar = {{1, 2}, {3, 4}};
    for (size_t i = 0; i < matrixPar.size(); ++i) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixPar[i].data()));
    }

    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(resPar.size());
  }
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(resPar[0], 1);
    ASSERT_EQ(resPar[1], 3);
    ASSERT_EQ(resSeq[0], 1);
    ASSERT_EQ(resSeq[1], 3);
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Main1) {
  std::vector<std::vector<int>> matrix;
  const int n = 10;
  const int m = 10;
  std::vector<int32_t> resPar(n);
  std::vector<int> res(n);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = getRandomMatrix_(n, m);
    for (size_t i = 0; i < matrix.size(); ++i) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    }

    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(resPar.size());
  }
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(resPar[i], -1);
    }
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Main2) {
  std::vector<std::vector<int>> matrix;
  const int n = 1000;
  const int m = 1000;
  std::vector<int32_t> resPar(n);
  std::vector<int> res(n);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = getRandomMatrix_(n, m);
    for (size_t i = 0; i < matrix.size(); ++i) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    }

    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(resPar.size());
  }
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(resPar[i], -1);
    }
  }
}
TEST(morozov_e_min_val_in_rows_matrix_MPI, Test_Main3) {
  std::vector<std::vector<int>> matrix;
  const int n = 10000;
  const int m = 10000;
  std::vector<int32_t> resPar(n);
  std::vector<int> res(n);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = getRandomMatrix_(n, m);
    for (size_t i = 0; i < matrix.size(); ++i) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    }

    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resPar.data()));
    taskDataPar->outputs_count.emplace_back(resPar.size());
  }
  morozov_e_min_val_in_rows_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(resPar[i], -1);
    }
  }
}
