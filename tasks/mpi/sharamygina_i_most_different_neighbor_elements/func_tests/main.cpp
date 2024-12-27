#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/map.hpp>
#include <random>
#include <vector>

#include "mpi/sharamygina_i_most_different_neighbor_elements/include/ops_mpi.hpp"

namespace sharamygina_i_most_different_neighbor_elements_mpi {
void generator(std::vector<int>& v) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = -1000 + gen() % 1000;
  }
}
}  // namespace sharamygina_i_most_different_neighbor_elements_mpi

TEST(sharamygina_i_most_different_neighbor_elements_mpi, wrong_test_1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_ans(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), false);
  }
}

TEST(sharamygina_i_most_different_neighbor_elements_mpi, wrong_test_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_ans(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), false);
  }
}

TEST(sharamygina_i_most_different_neighbor_elements_mpi, minimum_size) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(2);
  std::vector<int> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    sharamygina_i_most_different_neighbor_elements_mpi::generator(global_vec);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}
TEST(sharamygina_i_most_different_neighbor_elements_mpi, test_with_size_not_divide_2_3_4) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(7);
  std::vector<int> global_max(1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    sharamygina_i_most_different_neighbor_elements_mpi::generator(global_vec);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_max(1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sharamygina_i_most_different_neighbor_elements_mpi, test_with_size_divide_2_3_4) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(120);
  std::vector<int> global_max(1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    sharamygina_i_most_different_neighbor_elements_mpi::generator(global_vec);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_max(1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sharamygina_i_most_different_neighbor_elements_mpi, test_with_equal_elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(3, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(sharamygina_i_most_different_neighbor_elements_mpi, Test_with_no_random) {
  boost::mpi::communicator world;
  std::vector<int> global_diff(1, 0);
  std::vector<int> global_vec = {1, 10, 12, 34, 87, 90, 5, 15, 10, 30, 22, 101, 77, 89};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>(3, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}
