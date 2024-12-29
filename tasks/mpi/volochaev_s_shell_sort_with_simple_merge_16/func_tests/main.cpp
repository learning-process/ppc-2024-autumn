#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/volochaev_s_shell_sort_with_simple_merge_16/include/ops_mpi.hpp"

namespace volochaev_s_shell_sort_with_simple_merge_16_mpi {

void get_random_matrix(std::vector<int> &mat, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = dis(gen);
  }
}

}  // namespace volochaev_s_shell_sort_with_simple_merge_16_mpi

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_WA_val) {
  boost::mpi::communicator world;
  std::vector<int> global_A(100);

  ASSERT_ANY_THROW(volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, 90, -100));
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_0) {
  boost::mpi::communicator world;
  std::vector<int> global_A;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_negative_size_WA_val) {
  boost::mpi::communicator world;
  std::vector<int> global_A;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100, 0);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(-1);

    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    EXPECT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_small_mas) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(10);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 100);

    global_res.resize(10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(10, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_prime_size) {
  boost::mpi::communicator world;

  std::vector<int> global_A;

  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(13);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 100);

    global_res.resize(13, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(13, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_medium_size) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(20);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 200);

    global_res.resize(20, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(20, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_more_medium_size) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(100);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -1000, 1000);

    global_res.resize(100, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(100, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_square_size) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(121);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 100);

    global_res.resize(121, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(121, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_big_square_size) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(289);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 1000);

    global_res.resize(289, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(289, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_with_corner_case) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(1);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 100);

    global_res.resize(1, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}

TEST(volochaev_s_shell_sort_with_simple_merge_16_mpi, Test_reverse_mas) {
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_A.resize(1000);
    volochaev_s_shell_sort_with_simple_merge_16_mpi::get_random_matrix(global_A, -100, 100);
    std::sort(global_A.begin(), global_A.end());
    std::reverse(global_A.begin(), global_A.end());

    global_res.resize(1000, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(global_A.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_mpi taskParallel(taskDataPar);

  ASSERT_TRUE(taskParallel.validation());
  ASSERT_TRUE(taskParallel.pre_processing());
  ASSERT_TRUE(taskParallel.run());
  ASSERT_TRUE(taskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_res(1000, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_A.data()));
    taskDataSeq->inputs_count.emplace_back(global_A.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expected_res.data()));
    taskDataSeq->outputs_count.emplace_back(expected_res.size());

    // Create Task
    volochaev_s_shell_sort_with_simple_merge_16_mpi::Lab3_16_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_res, expected_res);
  }
}
