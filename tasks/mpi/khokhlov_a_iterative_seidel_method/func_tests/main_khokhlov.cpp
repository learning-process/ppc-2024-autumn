#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/khokhlov_a_iterative_seidel_method/include/ops_mpi_khokhlov.hpp"
#include "mpi/khokhlov_a_iterative_seidel_method/src/ops_mpi_khokhlov.cpp"

void getRandomSLAU(std::vector<double> &A, std::vector<double> &b, int N) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < N; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = rand() % 10 - 5;
        rowSum += std::abs(A[i * N + j]);
      }
    }
    A[i * N + i] = rowSum + (rand() % 5 + 1);
    b[i] = rand() % 20 - 10;
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_empty_matrix) {
  boost::mpi::communicator world;
  const int n = 0;
  const int maxiter = 10;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {};
  std::vector<double> b = {};
  std::vector<double> expect = {};
  std::vector<double> result = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(mpi_task.validation());
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_invalid_iter) {
  boost::mpi::communicator world;
  const int n = 2;
  const int maxiter = 0;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {1, 2, 3, 4};
  std::vector<double> b = {1, 2};
  std::vector<double> expect = {};
  std::vector<double> result = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(mpi_task.validation());
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_validation) {
  boost::mpi::communicator world;
  const int n = 2;
  const int maxiter = 10;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {1, 2, 3, 4};
  std::vector<double> b = {1, 2};
  std::vector<double> expect = {1, 1};
  std::vector<double> result = {};

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  }

  // crate task
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  ASSERT_TRUE(mpi_task.validation());
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_const_matrix_10x10) {
  boost::mpi::communicator world;
  const int n = 10;
  const int maxiter = 100;
  const double eps = 1e-8;

  // Create data
  std::vector<double> A(n * n, 1.0);
  std::vector<double> b(n, 1.0);
  std::vector<double> result(n, 1.0);

  getRandomSLAU(A, b, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  };
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  ASSERT_TRUE(mpi_task.validation());
  mpi_task.pre_processing();
  mpi_task.run();
  mpi_task.post_processing();

  ASSERT_EQ(result.size(), b.size());

  if (world.rank() == 0) {
    std::vector<double> result_seq(n, 0.0);
    // create task data
    std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataSeq->inputs_count.emplace_back(n);
    taskdataSeq->inputs_count.emplace_back(maxiter);
    taskdataSeq->inputs_count.emplace_back(eps);
    taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
    taskdataSeq->outputs_count.emplace_back(result_seq.size());

    // crate task
    khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seq_task(taskdataSeq);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();
    for (int i = 0; i < n; i++) {
      ASSERT_NEAR(result[i], result_seq[i], 1);
    }
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_const_matrix_20x20) {
  boost::mpi::communicator world;
  const int n = 20;
  const int maxiter = 100;
  const double eps = 1e-8;

  // Create data
  std::vector<double> A(n * n, 1.0);
  std::vector<double> b(n, 1.0);
  std::vector<double> result(n, 1.0);

  getRandomSLAU(A, b, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  };
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  ASSERT_TRUE(mpi_task.validation());
  mpi_task.pre_processing();
  mpi_task.run();
  mpi_task.post_processing();

  ASSERT_EQ(result.size(), b.size());

  if (world.rank() == 0) {
    std::vector<double> result_seq(n, 0.0);
    // create task data
    std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataSeq->inputs_count.emplace_back(n);
    taskdataSeq->inputs_count.emplace_back(maxiter);
    taskdataSeq->inputs_count.emplace_back(eps);
    taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
    taskdataSeq->outputs_count.emplace_back(result_seq.size());

    // crate task
    khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seq_task(taskdataSeq);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();
    for (int i = 0; i < n; i++) {
      ASSERT_NEAR(result[i], result_seq[i], 5 * 1e-1);
    }
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_const_matrix_50x50) {
  boost::mpi::communicator world;
  const int n = 50;
  const int maxiter = 100;
  const double eps = 1e-8;

  // Create data
  std::vector<double> A(n * n, 1.0);
  std::vector<double> b(n, 1.0);
  std::vector<double> result(n, 1.0);

  getRandomSLAU(A, b, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  };
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  ASSERT_TRUE(mpi_task.validation());
  mpi_task.pre_processing();
  mpi_task.run();
  mpi_task.post_processing();

  ASSERT_EQ(result.size(), b.size());

  if (world.rank() == 0) {
    std::vector<double> result_seq(n, 0.0);
    // create task data
    std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataSeq->inputs_count.emplace_back(n);
    taskdataSeq->inputs_count.emplace_back(maxiter);
    taskdataSeq->inputs_count.emplace_back(eps);
    taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
    taskdataSeq->outputs_count.emplace_back(result_seq.size());

    // crate task
    khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seq_task(taskdataSeq);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();
    for (int i = 0; i < n; i++) {
      ASSERT_NEAR(result[i], result_seq[i], 5 * 1e-1);
    }
  }
}

TEST(khokhlov_a_iterative_seidel_method_mpi, test_const_matrix100x100) {
  boost::mpi::communicator world;
  const int n = 100;
  const int maxiter = 150;
  const double eps = 1e-8;

  // Create data
  std::vector<double> A(n * n, 1.0);
  std::vector<double> b(n, 1.0);
  std::vector<double> result(n, 1.0);

  getRandomSLAU(A, b, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskdataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataMpi->inputs_count.emplace_back(n);
    taskdataMpi->inputs_count.emplace_back(maxiter);
    taskdataMpi->inputs_count.emplace_back(eps);
    taskdataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskdataMpi->outputs_count.emplace_back(result.size());
  };
  khokhlov_a_iterative_seidel_method_mpi::seidel_method_mpi mpi_task(taskdataMpi);
  ASSERT_TRUE(mpi_task.validation());
  mpi_task.pre_processing();
  mpi_task.run();
  mpi_task.post_processing();

  ASSERT_EQ(result.size(), b.size());

  if (world.rank() == 0) {
    std::vector<double> result_seq(n, 0.0);
    // create task data
    std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskdataSeq->inputs_count.emplace_back(n);
    taskdataSeq->inputs_count.emplace_back(maxiter);
    taskdataSeq->inputs_count.emplace_back(eps);
    taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
    taskdataSeq->outputs_count.emplace_back(result_seq.size());

    // crate task
    khokhlov_a_iterative_seidel_method_mpi::seidel_method_seq seq_task(taskdataSeq);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();
    for (int i = 0; i < n; i++) {
      ASSERT_NEAR(result[i], result_seq[i], 5 * 1e-1);
    }
  }
}