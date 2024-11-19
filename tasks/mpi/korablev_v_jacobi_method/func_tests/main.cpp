#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/korablev_v_jacobi_method/include/ops_mpi.hpp"

std::pair<std::vector<double>, std::vector<double>> generate_diagonally_dominant_matrix(int n, double min_val = -10.0,
                                                                                        double max_val = 10.0) {
  std::vector<double> A(n * n);
  std::vector<double> b(n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i * n + j] = dist(gen);
        row_sum += std::abs(A[i * n + j]);
      }
    }

    A[i * n + i] = row_sum + std::abs(dist(gen)) + 1.0;

    b[i] = dist(gen);
  }

  return {A, b};
}

void run_jacobi_test_for_matrix_size(size_t matrix_size) {
  boost::mpi::communicator world;

  auto [A_flat, b] = generate_diagonally_dominant_matrix(matrix_size);

  std::vector<double> x_parallel(matrix_size, 0.0);
  std::vector<double> x_sequential(matrix_size, 0.0);

  size_t matrix_size_copy = matrix_size;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataPar->inputs_count.emplace_back(A_flat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_parallel.data()));
    taskDataPar->outputs_count.emplace_back(x_parallel.size());
  }

  korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobi_parallel(taskDataPar);
  ASSERT_TRUE(jacobi_parallel.validation());
  jacobi_parallel.pre_processing();
  jacobi_parallel.run();
  jacobi_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_flat.data()));
    taskDataSeq->inputs_count.emplace_back(A_flat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(x_sequential.size());

    korablev_v_jacobi_method_mpi::JacobiMethodSequential jacobi_sequential(taskDataSeq);
    ASSERT_TRUE(jacobi_sequential.validation());
    jacobi_sequential.pre_processing();
    jacobi_sequential.run();
    jacobi_sequential.post_processing();
  }
  if (world.rank() == 0) {
    for (size_t i = 0; i < matrix_size; ++i) {
      ASSERT_NEAR(x_parallel[i], x_sequential[i], 1e-3);
    }
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}
TEST(korablev_v_jacobi_method_mpi, test_matrix_2x2) { run_jacobi_test_for_matrix_size(2); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_3x3) { run_jacobi_test_for_matrix_size(3); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_4x4) { run_jacobi_test_for_matrix_size(4); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_5x5) { run_jacobi_test_for_matrix_size(5); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_8x8) { run_jacobi_test_for_matrix_size(8); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_16x16) { run_jacobi_test_for_matrix_size(16); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_32x32) { run_jacobi_test_for_matrix_size(32); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_100x100) { run_jacobi_test_for_matrix_size(100); }
TEST(korablev_v_jacobi_method_mpi, test_matrix_1000x1000) { run_jacobi_test_for_matrix_size(1000); }

TEST(korablev_v_jacobi_method_mpi, invalid_input_count) {
  boost::mpi::communicator world;
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  if (world.rank() == 0) {
    korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(korablev_v_jacobi_method_mpi, invalid_output_count) {
  boost::mpi::communicator world;
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {4.0, 1.0, 2.0, 3.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  if (world.rank() == 0) {
    korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(korablev_v_jacobi_method_mpi, non_diagonally_dominant_matrix) {
  boost::mpi::communicator world;
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {1.0, 2.0, 2.0, 1.0};
  std::vector<double> vector_data = {1.0, 2.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  if (world.rank() == 0) {
    korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(korablev_v_jacobi_method_mpi, zero_on_diagonal) {
  boost::mpi::communicator world;
  const size_t matrix_size = 2;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {0.0, -1.0, -2.0, 0.0};
  std::vector<double> vector_data = {3.0, 4.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(korablev_v_jacobi_method_mpi, invalid_matrix_size) {
  boost::mpi::communicator world;
  const size_t matrix_size = 0;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {0.0, -1.0, -2.0, 0.0};
  std::vector<double> vector_data = {3.0, 4.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  if (world.rank() == 0) {
    korablev_v_jacobi_method_mpi::JacobiMethodParallel jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}