// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/drozhdinov_d_mult_matrix_fox/include/ops_mpi.hpp"
using namespace drozhdinov_d_mult_matrix_fox_mpi;
namespace drozhdinov_d_mult_matrix_fox_mpi {
std::vector<double> MatrixMult(const std::vector<double> &A, const std::vector<double> &B, int k, int l, int n) {
  std::vector<double> result(k * n, 0.0);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < l; p++) {
        result[i * n + j] += A[i * l + p] * B[p * n + j];
      }
    }
  }

  return result;
}

std::vector<double> getRandomMatrix(int sz, int lbound, int rbound) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  std::uniform_int_distribution<int> dist(lbound, rbound);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace drozhdinov_d_mult_matrix_fox_mpi

TEST(drozhdinov_d_mult_matrix_fox_MPI, 2x3_3x2Test) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 2;
  int l = 3;
  int m = 3;
  int n = 2;
  std::vector<double> A = {1, 2, 3, 4, 5, 6};
  std::vector<double> B = {7, 8, 9, 10, 11, 12};
  std::vector<double> expres_par(4);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(4);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres[i], expres_par[i]);
      EXPECT_DOUBLE_EQ(expres[i], expres_seq[i]);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, Random100Test) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 100;
  int l = 100;
  int m = 100;
  int n = 100;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(k * n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres[i], expres_par[i]);
      EXPECT_DOUBLE_EQ(expres[i], expres_seq[i]);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, Random150Test) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 150;
  int l = 150;
  int m = 150;
  int n = 150;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(k * n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres[i], expres_par[i]);
      EXPECT_DOUBLE_EQ(expres[i], expres_seq[i]);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, Random50_100x100_50Test) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 50;
  int l = 100;
  int m = 100;
  int n = 50;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(k * n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres[i], expres_par[i]);
      EXPECT_DOUBLE_EQ(expres[i], expres_seq[i]);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, WrongValidation1) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 2;
  int l = 10;
  int m = 11;
  int n = 2;  // A cols not eq B rows
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  // std::vector<double> expres = MatrixMult(A, B, k, l, n);
  //  Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(4);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, WrongValidation2) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 2;
  int l = 10;
  int m = 10;
  int n = 2;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  // std::vector<double> expres = MatrixMult(A, B, k, l, n);
  //  Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    // only 1 matrix given
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(4);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    // only one matrix given
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, NotSquareProcesses) {
  boost::mpi::communicator world;
  int k = 2;
  int l = 3;
  int m = 3;
  int n = 2;
  std::vector<double> A = {1, 2, 3, 4, 5, 6};
  std::vector<double> B = {7, 8, 9, 10, 11, 12};
  std::vector<double> expres_par(4);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.size() == 2 || world.size() == 3) {
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
      taskDataPar->inputs_count.emplace_back(k);
      taskDataPar->inputs_count.emplace_back(l);
      taskDataPar->inputs_count.emplace_back(m);
      taskDataPar->inputs_count.emplace_back(n);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
      taskDataPar->outputs_count.emplace_back(k);
      taskDataPar->outputs_count.emplace_back(n);
      drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
      ASSERT_EQ(testMpiTaskParallel.validation(), true);
      testMpiTaskParallel.pre_processing();
      testMpiTaskParallel.run();
      testMpiTaskParallel.post_processing();
    }
    if (world.rank() == 0) {
      // Create data
      std::vector<double> expres_seq(4);
      // Create TaskData
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
      taskDataSeq->inputs_count.emplace_back(k);
      taskDataSeq->inputs_count.emplace_back(l);
      taskDataSeq->inputs_count.emplace_back(m);
      taskDataSeq->inputs_count.emplace_back(n);
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
      taskDataSeq->outputs_count.emplace_back(k);
      taskDataSeq->outputs_count.emplace_back(n);

      // Create Task
      drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();
      ASSERT_EQ(expres, expres_par);
      ASSERT_EQ(expres, expres_seq);
    }
  } else {
    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
      taskDataPar->inputs_count.emplace_back(k);
      taskDataPar->inputs_count.emplace_back(l);
      taskDataPar->inputs_count.emplace_back(m);
      taskDataPar->inputs_count.emplace_back(n);
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
      taskDataPar->outputs_count.emplace_back(k);
      taskDataPar->outputs_count.emplace_back(n);
    }

    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      // Create data
      std::vector<double> expres_seq(4);

      // Create TaskData
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
      taskDataSeq->inputs_count.emplace_back(k);
      taskDataSeq->inputs_count.emplace_back(l);
      taskDataSeq->inputs_count.emplace_back(m);
      taskDataSeq->inputs_count.emplace_back(n);
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
      taskDataSeq->outputs_count.emplace_back(k);
      taskDataSeq->outputs_count.emplace_back(n);

      // Create Task
      drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();
      ASSERT_EQ(expres, expres_par);
      ASSERT_EQ(expres, expres_seq);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, WrongValidation4) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 2;
  int l = 10;
  int m = 10;
  int n = 2;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  // std::vector<double> expres = MatrixMult(A, B, k, l, n);
  //  Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k + 1);  // outputs_count wrong
    taskDataPar->outputs_count.emplace_back(n);
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(4);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k + 1);  // outputs_count wrong
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_MPI, EmptyTest) {
  boost::mpi::communicator world;
  if (world.size() != 1 && world.size() != 4) {
    ASSERT_TRUE(true);
    return;
  }
  int k = 0;
  int l = 0;
  int m = 0;
  int n = 0;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> expres_par;
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(l);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(k);
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(expres, expres_par);
    ASSERT_EQ(expres, expres_seq);
  }
}