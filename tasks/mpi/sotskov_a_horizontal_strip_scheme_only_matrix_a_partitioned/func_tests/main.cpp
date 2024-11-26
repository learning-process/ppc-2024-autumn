#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_mpi.hpp"

namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi {

void get_random_matrix(std::vector<int> &mat) {
  std::random_device dev;
  std::mt19937 gen(dev());

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = gen() % 10 - 5;
  }
}

}  // namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, InitializationWithEmptyInputs) {
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

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, InvalidTaskWithPartialInputs) {
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

  auto taskParallel =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle>(
          taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
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
  }

  auto taskParallel =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle>(
          taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
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
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B);

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
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_A);
    sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::get_random_matrix(global_B);

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