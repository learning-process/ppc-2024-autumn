#include <gtest/gtest.h>

#include "mpi/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_mpi.hpp"
#include "seq/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_seq.hpp"

template <typename DataType>
static std::vector<DataType> generateRandomValues(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_fixed_values) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 3;
  size_t k = 3;
  size_t n = 2;
  std::vector<DataType> A = {1, 2, 3, 4, 5, 6, 4, 5, 6};
  std::vector<DataType> B = {7, 8, 9, 10, 11, 12};
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_sq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_sq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_sq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_negative_fixed_values) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 2;
  size_t k = 3;
  size_t n = 2;
  std::vector<DataType> A = {-3, -2, -3, -6, -9, -8};
  std::vector<DataType> B = {-7, -2, -1, -10, -3, -12};
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_random_values) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 3;
  size_t k = 4;
  size_t n = 2;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_float_values) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = float;
  size_t m = 3;
  size_t k = 3;
  size_t n = 3;
  std::vector<DataType> A = generateRandomValues<float>(m * k);
  std::vector<DataType> B = generateRandomValues<float>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_zero_matrix) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 4;
  size_t k = 4;
  size_t n = 4;
  std::vector<DataType> A(m * k, 0);
  std::vector<DataType> B(k * n, 0);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_double_values) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = double;
  size_t m = 4;
  size_t k = 2;
  size_t n = 3;
  std::vector<DataType> A = generateRandomValues<double>(m * k);
  std::vector<DataType> B = generateRandomValues<double>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_4x8x16) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 4;
  size_t k = 8;
  size_t n = 16;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_7x3x13) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 7;
  size_t k = 3;
  size_t n = 13;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_8x7x23) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 8;
  size_t k = 7;
  size_t n = 23;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_11x10x9) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  using DataType = int32_t;
  size_t m = 11;
  size_t k = 10;
  size_t n = 9;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C_par(m * n, 0);
  std::vector<DataType> C_seq(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C_par.size(), m * n);
  ASSERT_EQ(C_seq.size(), m * n);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_par.data()));

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(k);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_seq.data()));
  }

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (rank == 0) {
    moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> taskSequential(taskDataSeq);

    ASSERT_TRUE(taskSequential.validation());
    ASSERT_TRUE(taskSequential.pre_processing());
    ASSERT_TRUE(taskSequential.run());
    ASSERT_TRUE(taskSequential.post_processing());
    EXPECT_EQ(C_par, C_seq);
  }
}