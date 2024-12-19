#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_mpi.hpp"

namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi {

void get_random_matrix(std::vector<int> &mat, int A, int B) {
  if (A > B) {
    throw std::invalid_argument("Invalid range: A must be less than or equal to B.");
  }
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(A, B);

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = dist(gen);
  }
}

}  // namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, TestVariousRanges) {
  boost::mpi::communicator world;
  std::vector<int> global_A(100);

  ASSERT_NO_THROW(
      sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -50, 50));
  ASSERT_NO_THROW(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, 0, 100));
  ASSERT_NO_THROW(
      sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -100, -10));

  ASSERT_ANY_THROW(
      sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, 90, -100));
  ASSERT_ANY_THROW(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, 10, 5));
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, InitializationWithEmptyInputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int result_size = 0;
  if (world.rank() == 0) {
    global_res.resize(result_size, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(0);
    taskDataSeq->inputs_count.emplace_back(0);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(0);
    taskDataSeq->inputs_count.emplace_back(0);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataSeq->outputs_count.emplace_back(global_res.size());

    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, InvalidTaskWithPartialInputs) {
  boost::mpi::communicator world;
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int result_size = 0;
  if (world.rank() == 0) {
    global_A.resize(100, 0);
    global_res.resize(result_size, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(25);
    taskDataSeq->inputs_count.emplace_back(4);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(0);
    taskDataSeq->inputs_count.emplace_back(0);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataSeq->outputs_count.emplace_back(global_res.size());

    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, InvalidTaskWithMismatchedDimensions) {
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

    auto taskParallel =
        std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential>(
            taskDataPar);
    EXPECT_FALSE(taskParallel->validation());
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithSquareMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(16);
    global_B.resize(16);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -10, 10);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 20);

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

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

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

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithRectangularMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(16);
    global_B.resize(8);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -50, 50);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 10);

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

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

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

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithPowersOfTen) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(10 * 10);
    global_B.resize(10 * 10);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -100, 100);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 50);

    global_res.resize(10 * 10, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(10);
    taskDataPar->inputs_count.emplace_back(10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(10);
    taskDataPar->inputs_count.emplace_back(10);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(10 * 10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(10);
    taskDataSeq->inputs_count.emplace_back(10);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(10);
    taskDataSeq->inputs_count.emplace_back(10);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithPrimeNumbers) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(7 * 7);
    global_B.resize(7 * 7);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -10, 10);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 20);

    global_res.resize(7 * 7, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(7);
    taskDataPar->inputs_count.emplace_back(7);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(7);
    taskDataPar->inputs_count.emplace_back(7);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(7 * 7, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(7);
    taskDataSeq->inputs_count.emplace_back(7);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(7);
    taskDataSeq->inputs_count.emplace_back(7);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithPowersOfTwo) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(16 * 16);
    global_B.resize(16 * 16);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -10, 10);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 20);

    global_res.resize(16 * 16, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(16);
    taskDataPar->inputs_count.emplace_back(16);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(16);
    taskDataPar->inputs_count.emplace_back(16);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(16 * 16, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(16);
    taskDataSeq->inputs_count.emplace_back(16);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(16);
    taskDataSeq->inputs_count.emplace_back(16);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithLargeMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100 * 100);
    global_B.resize(100 * 100);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -500, 500);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 1000);

    global_res.resize(100 * 100, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(100);
    taskDataPar->inputs_count.emplace_back(100);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(100);
    taskDataPar->inputs_count.emplace_back(100);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(100 * 100, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(100);
    taskDataSeq->inputs_count.emplace_back(100);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(100);
    taskDataSeq->inputs_count.emplace_back(100);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithRandomSizedMatrices) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(240 * 120);
    global_B.resize(120 * 240);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -200, 200);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 500);

    global_res.resize(240 * 240, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(240);
    taskDataPar->inputs_count.emplace_back(120);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(120);
    taskDataPar->inputs_count.emplace_back(240);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(240 * 240, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(240);
    taskDataSeq->inputs_count.emplace_back(120);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(120);
    taskDataSeq->inputs_count.emplace_back(240);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, ValidationAndExecutionWithSingleElementMatrix) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(1);
    global_B.resize(1);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A, -10, 10);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B, 0, 20);

    global_res.resize(1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_B.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential testMpiTaskSequential(
        taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}