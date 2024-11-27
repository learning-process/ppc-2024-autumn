#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/kavtorev_d_jacobi_method/include/ops_mpi.hpp"

namespace kavtorev_d_jacobi_method_mpi {
double calculate_residual(const std::vector<double>& matrix, const std::vector<double>& solution,
                          const std::vector<double>& rhs, size_t matrix_size) {
  std::vector<double> calculated_rhs(matrix_size, 0.0);
  double residual_sum = 0.0;

  for (size_t row = 0; row < matrix_size; ++row) {
    for (size_t col = 0; col < matrix_size; ++col) {
      calculated_rhs[row] += matrix[row * matrix_size + col] * solution[col];
    }
  }

  for (size_t i = 0; i < matrix_size; ++i) {
    double difference = calculated_rhs[i] - rhs[i];
    residual_sum += difference * difference;
  }

  return std::sqrt(residual_sum);
}

std::pair<std::vector<double>, std::vector<double>> get_random_diagonally_matrix(int matrix_size,
                                                                                 double min_val = -10.0,
                                                                                 double max_val = 10.0) {
  std::vector<double> matrix(matrix_size * matrix_size);
  std::vector<double> rhs(matrix_size);

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_real_distribution<> distribution(min_val, max_val);

  for (int row = 0; row < matrix_size; ++row) {
    double off_diagonal_sum = 0.0;

    for (int col = 0; col < matrix_size; ++col) {
      if (row != col) {
        matrix[row * matrix_size + col] = distribution(generator);
        off_diagonal_sum += std::abs(matrix[row * matrix_size + col]);
      }
    }

    matrix[row * matrix_size + row] = off_diagonal_sum + std::abs(distribution(generator)) + 1.0;

    rhs[row] = distribution(generator);
  }

  return {matrix, rhs};
}
}  // namespace kavtorev_d_jacobi_method_mpi

void run_jacobi_method(size_t matrix_size) {
  boost::mpi::communicator world;

  auto [matrix_A, rsh] = kavtorev_d_jacobi_method_mpi::get_random_diagonally_matrix(matrix_size);

  std::vector<double> x_parallel(matrix_size, 0.0);
  std::vector<double> x_sequential(matrix_size, 0.0);

  size_t matrix_size_copy = matrix_size;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_A.data()));
    taskDataPar->inputs_count.emplace_back(matrix_A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rsh.data()));
    taskDataPar->inputs_count.emplace_back(rsh.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_parallel.data()));
    taskDataPar->outputs_count.emplace_back(x_parallel.size());
  }

  kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobi_parallel(taskDataPar);
  ASSERT_TRUE(jacobi_parallel.validation());
  jacobi_parallel.pre_processing();
  jacobi_parallel.run();
  jacobi_parallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(matrix_A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rsh.data()));
    taskDataSeq->inputs_count.emplace_back(rsh.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_sequential.data()));
    taskDataSeq->outputs_count.emplace_back(x_sequential.size());

    kavtorev_d_jacobi_method_mpi::JacobiMethodSequentialTask jacobi_sequential(taskDataSeq);
    ASSERT_TRUE(jacobi_sequential.validation());
    jacobi_sequential.pre_processing();
    jacobi_sequential.run();
    jacobi_sequential.post_processing();
  }
  if (world.rank() == 0) {
    double residual_parallel =
        kavtorev_d_jacobi_method_mpi::calculate_residual(matrix_A, x_parallel, rsh, matrix_size);
    double residual_sequential =
        kavtorev_d_jacobi_method_mpi::calculate_residual(matrix_A, x_sequential, rsh, matrix_size);

    ASSERT_LT(residual_parallel, 1e-3) << "Parallel solution did not converge within tolerance.";
    ASSERT_LT(residual_sequential, 1e-3) << "Sequential solution did not converge within tolerance.";
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, known_solution) {
  const size_t matrix_size = 3;
  size_t matrix_size_copy = matrix_size;
  std::vector<double> matrix(matrix_size * matrix_size);
  std::vector<double> rhs(matrix_size);

  for (size_t row = 0; row < matrix_size; ++row) {
    for (size_t col = 0; col < matrix_size; ++col) {
      if (row != col) {
        matrix[row * matrix_size + col] = 0.0;
      }
    }

    matrix[row * matrix_size + row] = 5.0;

    rhs[row] = 5.0;
  }

  boost::mpi::communicator world;

  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->inputs_count.emplace_back(rhs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_TRUE(jacobiTaskParallel.validation());
    jacobiTaskParallel.pre_processing();
    jacobiTaskParallel.run();
    jacobiTaskParallel.post_processing();
  }

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 1.0);
    ASSERT_EQ(out[1], 1.0);
    ASSERT_EQ(out[2], 1.0);
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, identity_matrix) {
  boost::mpi::communicator world;
  const size_t matrix_size = 3;
  size_t matrix_size_copy = matrix_size;
  std::vector<double> matrix_data = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> vector_data = {1.0, 2.0, 3.0};
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_data.data()));
    taskDataPar->inputs_count.emplace_back(matrix_data.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_data.data()));
    taskDataPar->inputs_count.emplace_back(vector_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_TRUE(jacobiTaskParallel.validation());
    jacobiTaskParallel.pre_processing();
    jacobiTaskParallel.run();
    jacobiTaskParallel.post_processing();
  }

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 1.0);
    ASSERT_EQ(out[1], 2.0);
    ASSERT_EQ(out[2], 3.0);
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, invalid_input) {
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, invalid_output) {
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, non_diagonally_dominant_matrix) {
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, diagonal_0) {
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, invalid_matrix_size) {
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, singular_matrix) {
  boost::mpi::communicator world;
  const size_t matrix_size = 0;
  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> matrix_data = {1.0, 2.0, 2.0, 4.0};
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
    kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask jacobiTaskParallel(taskDataPar);
    ASSERT_FALSE(jacobiTaskParallel.validation());
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(kavtorev_d_jacobi_method_mpi, matrix_2x2) { run_jacobi_method(2); }

TEST(kavtorev_d_jacobi_method_mpi, matrix_3x3) { run_jacobi_method(3); }

TEST(kavtorev_d_jacobi_method_mpi, matrix_5x5) { run_jacobi_method(5); }

TEST(kavtorev_d_jacobi_method_mpi, matrix_10x10) { run_jacobi_method(10); }

TEST(kavtorev_d_jacobi_method_mpi, matrix_100x100) { run_jacobi_method(100); }

TEST(kavtorev_d_jacobi_method_mpi, matrix_500x500) { run_jacobi_method(500); }