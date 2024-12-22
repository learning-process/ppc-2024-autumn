// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/ivanov_m_gauss_horizontal/include/ops_mpi.hpp"

namespace ivanov_m_gauss_horizontal_mpi {
std::vector<double> GenSolution(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> solution(size, 0);

  for (int i = 0; i < size; i++) {
    solution[i] = static_cast<double>(generator(gen));  // generating random coefficient in range [-2, 2]
  }
  return solution;
}

std::vector<double> GenMatrix(const std::vector<double> &solution) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> extended_matrix;
  int size = static_cast<int>(solution.size());

  // generate identity matrix
  for (int row = 0; row < size; row++) {
    for (int column = 0; column < size; column++) {
      if (row == column) {
        extended_matrix.push_back(1);
      } else {
        extended_matrix.push_back(0);
      }
    }
    extended_matrix.push_back(solution[row]);
  }

  // saturation left triangle
  for (int row = 1; row < size; row++) {
    for (int column = 0; column < row; column++) {
      extended_matrix[get_linear_index(row, column, size + 1)] +=
          extended_matrix[get_linear_index(row - 1, column, size + 1)];
    }
    extended_matrix[get_linear_index(row, size, size + 1)] +=
        extended_matrix[get_linear_index(row - 1, size, size + 1)];
  }

  // saturation of matrix by random numbers
  for (int row = size - 1; row > 0; row--) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row - 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  // saturation of matrix by random numbers
  for (int row = 0; row < size - 1; row++) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row + 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  return extended_matrix;
}
}  // namespace ivanov_m_gauss_horizontal_mpi

TEST(ivanov_m_gauss_horizontal_mpi_func_test, validation_false_test_inputs_sizes) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, validation_false_test_inputs_count_sizes) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, validation_false_test_outputs_sizes) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, validation_false_test_outputs_count_sizes) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, validation_false_test_inputs_nullptr) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix;
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    EXPECT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, pre_processing_false_test_size_of_matrix_0) {
  boost::mpi::communicator world;
  int n = 0;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, pre_processing_false_test_size_of_matrix_more) {
  boost::mpi::communicator world;
  int n = 3;
  std::vector<double> matrix = {1, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, pre_processing_false_test_determinant_is_zero_because_rows_are_zero) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {0, 0, 2, 0, 1, 3};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, pre_processing_false_test_determinant_is_counted_and_equals_zero) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {1, 1, 1, 1, 1, 1};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), false);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, pre_processing_true_test_determinant_is_not_zero) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> matrix = {2, 0, 1, 0, 4, 1};
  std::vector<double> ans = {2, 3};
  std::vector<double> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  if (world.rank() == 0) {
    ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), true);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, run_random_matrix_size_2) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_seq(n, 0);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, run_random_matrix_size_5) {
  boost::mpi::communicator world;
  int n = 5;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_seq(n, 0);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, run_random_matrix_size_10) {
  boost::mpi::communicator world;
  int n = 10;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_seq(n, 0);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, run_random_matrix_size_100) {
  boost::mpi::communicator world;
  int n = 100;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_seq(n, 0);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-3);
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_func_test, run_random_matrix_size_300) {
  boost::mpi::communicator world;
  int n = 300;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_seq(n, 0);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }

  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-3);
  }
}