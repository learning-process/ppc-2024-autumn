#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

void get_random_matrix(std::vector<int> &mat, int min_val, int max_val) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(min_val, max_val);

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = dis(gen);  // Generate random number in the specified range
  }
}

bool is_prime(int num) {
  if (num <= 1) return false;
  for (int i = 2; i <= std::sqrt(num); ++i) {
    if (num % i == 0) return false;
  }
  return true;
}

void get_matrix_with_primes(std::vector<int> &mat, size_t rows, size_t cols, int min_val, int max_val) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min_val, max_val);

  for (size_t i = 0; i < rows * cols; ++i) {
    mat[i] = dist(gen);

    // Ensure some of the values are prime numbers
    if (i % 10 == 0) {  // Arbitrary condition to insert prime numbers
      mat[i] = 11;      // Insert a prime number at every 10th position (for example)
    }
  }
}
}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, InitializationWithEmptyInputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, InvalidTaskWithPartialInputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(25);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, InvalidTaskWithMismatchedDimensions) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int m = 0;

  if (world.rank() == 0) {
    global_res.resize(m, 0);
    global_A.resize(100, 0);
    global_B.resize(3, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(25);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto taskParallel = std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, ValidationAndExecutionWithSquareMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(16);
    global_B.resize(16);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_A, 5, 10);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_B, 5, 10);

    global_res.resize(16, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(16, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(4);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(4);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, ValidationAndExecutionWithRectangularMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(16);
    global_B.resize(8);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_A, 5, 10);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_B, 5, 10);

    global_res.resize(8, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(4);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(8, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(4);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(2);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, ValidationWithZeroMatrix) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = {0, 0, 0, 0};
    global_B = {2, 3, 4, 5};
    global_res.resize(4, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res = {0, 0, 0, 0};
    EXPECT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, MismatchedDimensionsValidation) {
  boost::mpi::communicator world;

  std::vector<int> global_A = {1, 2, 3};
  std::vector<int> global_B = {4, 5};
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(3);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(0);
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel.validation());
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, SmallMatrixMultiplication) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = {1, 2, 3, 4};
    global_B = {5, 6, 7, 8};
    global_res.resize(4, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res = {19, 22, 43, 50};
    EXPECT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, LargeSquareMatrix) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(10000);
    global_B.resize(10000);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_A, 5, 10);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_B, 5, 10);
    global_res.resize(10000, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(100);
    taskDataPar->inputs_count.emplace_back(100);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(100);
    taskDataPar->inputs_count.emplace_back(100);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, ValidationAndExecutionWithVector) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = {1, 2, 3};
    global_B = {4};
    global_res.resize(3, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res = {4, 8, 12};
    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, ValidationAndExecutionWithIdentityMatrix) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A = {1, 0, 0, 1};
    global_B = {2, 3, 4, 5};
    global_res.resize(4, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res = {2, 3, 4, 5};
    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, LargeRectangularMatrix) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    size_t rows_A = 500;
    size_t cols_A = 1000;
    size_t rows_B = 1000;
    size_t cols_B = 200;

    global_A.resize(rows_A * cols_A);
    global_B.resize(rows_B * cols_B);

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_A, -5, 5);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_B, -5, 5);

    global_res.resize(rows_A * cols_B, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(500);
    taskDataPar->inputs_count.emplace_back(1000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(1000);
    taskDataPar->inputs_count.emplace_back(200);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, LargeRectangularMatrixWithPrimes) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    size_t rows_A = 500;
    size_t cols_A = 1000;
    size_t rows_B = 1000;
    size_t cols_B = 200;

    global_A.resize(rows_A * cols_A);
    global_B.resize(rows_B * cols_B);

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_matrix_with_primes(global_A, rows_A, cols_A, -5, 5);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_matrix_with_primes(global_B, rows_B, cols_B, -5, 5);

    global_res.resize(rows_A * cols_B, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(500);
    taskDataPar->inputs_count.emplace_back(1000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(1000);
    taskDataPar->inputs_count.emplace_back(200);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    bool found_prime = false;
    for (size_t i = 0; i < global_res.size(); ++i) {
      if (kalinin_d_matrix_mult_hor_a_vert_b_mpi::is_prime(global_res[i])) {
        found_prime = true;
        break;
      }
    }
    ASSERT_TRUE(found_prime);
  }
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_mpi, LargeRectangularMatrixWithPrimeDimensions) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    size_t rows_A = 5;
    size_t cols_A = 11;
    size_t rows_B = 11;
    size_t cols_B = 7;

    global_A.resize(rows_A * cols_A);
    global_B.resize(rows_B * cols_B);

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_A, -5, 5);
    kalinin_d_matrix_mult_hor_a_vert_b_mpi::get_random_matrix(global_B, -5, 5);

    global_res.resize(rows_A * cols_B, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(5);
    taskDataPar->inputs_count.emplace_back(11);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(11);
    taskDataPar->inputs_count.emplace_back(7);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());
}
