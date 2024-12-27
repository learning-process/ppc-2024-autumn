#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/veliev_e_jacobi_method/include/ops_mpi.hpp"

TEST(veliev_e_jacobi_method_mpi, Test_4x4_system) {
  boost::mpi::communicator world;

  const uint32_t systemSize = 4;
  std::vector<double> matrix = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 3};
  std::vector<double> rhs = {15, 10, 10, 10};
  std::vector<double> solution(systemSize, 0.0);
  double epsilon = 1e-6;

  std::vector<double> resultMPI(systemSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(systemSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(resultMPI.data()));
    taskDataMPI->outputs_count.push_back(resultMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI mpiTask(taskDataMPI);

  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  std::vector<double> matrixVectorProduct(systemSize, 0.0);
  for (uint32_t i = 0; i < systemSize; ++i) {
    for (uint32_t j = 0; j < systemSize; ++j) {
      matrixVectorProduct[i] += matrix[i * systemSize + j] * resultMPI[j];
    }
  }
  std::vector<double> result(systemSize, 0.0);
  for (uint32_t i = 0; i < systemSize; ++i) {
    result[i] = std::abs(matrixVectorProduct[i] - rhs[i]);
  }

  if (world.rank() == 0) {
    std::vector<double> resultSeq(systemSize, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.push_back(systemSize);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(resultSeq.data()));
    taskDataSeq->outputs_count.push_back(resultSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq seqTask(taskDataSeq);

    ASSERT_TRUE(seqTask.validation());
    seqTask.pre_processing();
    seqTask.run();
    seqTask.post_processing();

    std::vector<double> matrixVectorProductSeq(systemSize, 0.0);
    for (uint32_t i = 0; i < systemSize; ++i) {
      for (uint32_t j = 0; j < systemSize; ++j) {
        matrixVectorProductSeq[i] += matrix[i * systemSize + j] * resultSeq[j];
      }
    }
    std::vector<double> resultSeqFinal(systemSize, 0.0);
    for (uint32_t i = 0; i < systemSize; ++i) {
      resultSeqFinal[i] = std::abs(matrixVectorProductSeq[i] - rhs[i]);
    }

    for (uint32_t i = 0; i < systemSize; ++i) {
      ASSERT_LT(result[i], 1.1e-6);
    }
    for (uint32_t i = 0; i < systemSize; ++i) {
      ASSERT_LT(resultSeqFinal[i], 1.1e-6);
    }
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_incorrect_system_with_zero_diagonal) {
  boost::mpi::communicator world;

  const uint32_t systemSize = 3;
  std::vector<double> matrix = {0, -1, 0, -1, 0, -1, 0, -1, 0};
  std::vector<double> rhs = {30, 20, 10};
  std::vector<double> solution(systemSize, 0.0);
  double epsilon = 1e-6;

  std::vector<double> resultMPI(systemSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(systemSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(resultMPI.data()));
    taskDataMPI->outputs_count.push_back(resultMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI mpiTask(taskDataMPI);

  if (world.rank() == 0)
    ASSERT_FALSE(mpiTask.validation());
  else
    ASSERT_TRUE(mpiTask.validation());

  if (world.rank() == 0) {
    std::vector<double> resultSeq(systemSize, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.push_back(systemSize);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(resultSeq.data()));
    taskDataSeq->outputs_count.push_back(resultSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq seqTask(taskDataSeq);

    ASSERT_FALSE(seqTask.validation());
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_empty_system) {
  boost::mpi::communicator world;

  const uint32_t systemSize = 0;
  std::vector<double> matrix = {};
  double epsilon = 1e-6;
  std::vector<double> rhs = {};
  std::vector<double> solution(systemSize, 0.0);

  std::vector<double> resultMPI(systemSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(systemSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(resultMPI.data()));
    taskDataMPI->outputs_count.push_back(resultMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI mpiTask(taskDataMPI);

  if (world.rank() == 0)
    ASSERT_FALSE(mpiTask.validation());
  else
    ASSERT_EQ(true, true);

  if (world.rank() == 0) {
    std::vector<double> resultSeq(systemSize, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.push_back(systemSize);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(resultSeq.data()));
    taskDataSeq->outputs_count.push_back(resultSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq seqTask(taskDataSeq);

    ASSERT_FALSE(seqTask.validation());
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_10x10_system) {
  boost::mpi::communicator world;

  const uint32_t systemSize = 10;
  std::vector<double> matrix(systemSize * systemSize, -1);
  double epsilon = 1e-6;
  for (uint32_t i = 0; i < systemSize; i++) {
    matrix[i * systemSize + i] = 10;
  }

  std::vector<double> rhs(systemSize, 10);
  std::vector<double> solution(systemSize, 0.0);

  std::vector<double> resultMPI(systemSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(systemSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(resultMPI.data()));
    taskDataMPI->outputs_count.push_back(resultMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI mpiTask(taskDataMPI);

  ASSERT_TRUE(mpiTask.validation());
  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  std::vector<double> Ax(systemSize, 0.0);
  for (uint32_t i = 0; i < systemSize; ++i) {
    for (uint32_t j = 0; j < systemSize; ++j) {
      Ax[i] += matrix[i * systemSize + j] * resultMPI[j];
    }
  }
  std::vector<double> res(systemSize, 0.0);
  for (uint32_t i = 0; i < systemSize; ++i) {
    res[i] = std::abs(Ax[i] - rhs[i]);
  }

  if (world.rank() == 0) {
    std::vector<double> resultSeq(systemSize, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.push_back(systemSize);
    taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(resultSeq.data()));
    taskDataSeq->outputs_count.push_back(resultSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq seqTask(taskDataSeq);

    ASSERT_TRUE(seqTask.validation());
    seqTask.pre_processing();
    seqTask.run();
    seqTask.post_processing();

    std::vector<double> AxSeq(systemSize, 0.0);
    for (uint32_t i = 0; i < systemSize; ++i) {
      for (uint32_t j = 0; j < systemSize; ++j) {
        AxSeq[i] += matrix[i * systemSize + j] * resultSeq[j];
      }
    }
    std::vector<double> resSeq(systemSize, 0.0);
    for (uint32_t i = 0; i < systemSize; ++i) {
      resSeq[i] = std::abs(AxSeq[i] - rhs[i]);
    }

    for (uint32_t i = 0; i < systemSize; i++) {
      ASSERT_LT(res[i], 1.1e-5);
    }
    for (uint32_t i = 0; i < systemSize; i++) {
      ASSERT_LT(resSeq[i], 1.1e-5);
    }
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_negative_rhs) {
  boost::mpi::communicator world;

  int N = 4;
  std::vector<double> matrixA = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 3};
  std::vector<double> rhsB = {-15, -10, -10, -10};
  std::vector<double> initialGuessX = {0, 0, 0, 0};
  double epsilon = 1e-6;
  std::vector<double> resMPI(N, 0);
  std::vector<double> expected_result = {-5, -5, -5, -5};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> Ax(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Ax[i] += matrixA[i * N + j] * resMPI[j];
    }
  }
  std::vector<double> res(N, 0.0);
  for (int i = 0; i < N; ++i) {
    res[i] = abs(Ax[i] - rhsB[i]);
  }

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    std::vector<double> AxSeq(N, 0.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        AxSeq[i] += matrixA[i * N + j] * resSeq[j];
      }
    }
    std::vector<double> res2(N, 0.0);
    for (int i = 0; i < N; ++i) {
      res2[i] = abs(AxSeq[i] - rhsB[i]);
    }

    for (int i = 0; i < N; i++) ASSERT_LT(res[i], 1.1e-6);
    for (int i = 0; i < N; i++) ASSERT_LT(res2[i], 1.1e-6);
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_1x1_system) {
  boost::mpi::communicator world;

  int N = 1;
  std::vector<double> matrixA = {5};
  std::vector<double> rhsB = {100};
  std::vector<double> initialGuessX = {0};
  double epsilon = 1e-6;
  std::vector<double> resMPI(N, 0);
  std::vector<double> expected_result = {20};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI[0], resSeq[0]);
    ASSERT_EQ(resSeq[0], expected_result[0]);
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_negative_epsilon) {
  boost::mpi::communicator world;

  int N = 1;
  std::vector<double> matrixA = {5};
  std::vector<double> rhsB = {100};
  std::vector<double> initialGuessX = {0};
  double epsilon = -1e-6;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());

  } else {
    ASSERT_TRUE(testMpiTaskParallel.validation());
  }

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(N);

    veliev_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(veliev_e_jacobi_method_mpi, Test_not_single_solution) {
  boost::mpi::communicator world;

  int N = 4;
  std::vector<double> matrixA = {4, -1, 0, 0, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1};
  std::vector<double> rhsB = {0, 0, 0, 0};
  std::vector<double> initialGuessX = {0, 0, 0, 0};
  double epsilon = 1e-6;
  std::vector<double> resMPI(N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  veliev_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0)
    ASSERT_FALSE(testMpiTaskParallel.validation());
  else
    ASSERT_EQ(true, true);

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhsB.data()));  // "rshB" -> "rhsB"
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(initialGuessX.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    veliev_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}
