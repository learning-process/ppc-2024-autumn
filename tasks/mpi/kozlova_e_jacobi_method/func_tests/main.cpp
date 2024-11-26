#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kozlova_e_jacobi_method/include/ops_mpi.hpp"

TEST(kozlova_e_jacobi_method_mpi, Test_4x4_system) {
  boost::mpi::communicator world;

  int N = 4;
  std::vector<double> A = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 3};

  std::vector<double> B = {15, 10, 10, 10};
  std::vector<double> X = {0, 0, 0, 0};

  std::vector<double> resMPI(N, 0);
  std::vector<double> expected_result = {5, 5, 5, 5};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < resMPI.size(); i++) ASSERT_NEAR(resMPI[i], expected_result[i], 1e-2);
  }
}

TEST(kozlova_e_jacobi_method_mpi, Test_incorrect_system_with_zero_diagonal) {
  boost::mpi::communicator world;

  int N = 3;
  std::vector<double> A = {0, -1, 0, -1, 0, -1, 0, -1, 0};
  std::vector<double> B = {30, 20, 10};
  std::vector<double> X = {0, 0, 0};

  std::vector<double> resMPI(N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), false);

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), false);
  }
}

TEST(kozlova_e_jacobi_method_mpi, Test_empty_system) {
  boost::mpi::communicator world;

  int N = 0;
  std::vector<double> A = {};

  std::vector<double> B = {};
  std::vector<double> X = {};

  std::vector<double> resMPI(N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(kozlova_e_jacobi_method_mpi, Test_10x10_system) {
  boost::mpi::communicator world;

  int N = 10;
  std::vector<double> A(N * N, -1);

  for (int i = 0; i < N; i++) {
    A[i * N + i] = 10;
  }

  std::vector<double> B(N, 10);
  std::vector<double> X(N, 0);

  std::vector<double> resMPI(N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < resMPI.size(); i++) ASSERT_NEAR(resMPI[i], resSeq[i], 1e-4);
  }
}

TEST(kozlova_e_jacobi_method_mpi, Test_negative_B) {
  boost::mpi::communicator world;

  int N = 4;
  std::vector<double> A = {4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 4, -1, 0, 0, -1, 3};
  std::vector<double> B = {-15, -10, -10, -10};
  std::vector<double> X = {0, 0, 0, 0};

  std::vector<double> resMPI(N, 0);
  std::vector<double> expected_result = {-5, -5, -5, -5};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < resMPI.size(); i++) ASSERT_NEAR(resMPI[i], expected_result[i], 1e-6);
  }
}

TEST(kozlova_e_jacobi_method_mpi, Test_1x1_system) {
  boost::mpi::communicator world;

  int N = 1;
  std::vector<double> A = {5};
  std::vector<double> B = {100};
  std::vector<double> X = {0};

  std::vector<double> resMPI(N, 0);
  std::vector<double> expected_result = {20};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  kozlova_e_jacobi_method_mpi::MethodJacobiMPI testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> resSeq(N, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataSeq->inputs_count.emplace_back(N);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_jacobi_method_mpi::MethodJacobiSeq testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(resMPI[0], resSeq[0]);
    ASSERT_EQ(resSeq[0], expected_result[0]);
  }
}