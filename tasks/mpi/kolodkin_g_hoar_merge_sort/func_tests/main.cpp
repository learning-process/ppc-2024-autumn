// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolodkin_g_hoar_merge_sort/include/ops_mpi.hpp"

namespace kolodkin_g_random_function {
std::vector<int> create_random_vector(unsigned n, int min, int max) {
  std::vector<int> vector;
  for (unsigned i = 0; i < n; i++) {
    vector.push_back(min + rand() % max);
  }
  return vector;
}
};  // namespace kolodkin_g_random_function

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_vector_with_one_elems) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;

  if (world.rank() == 0) {
    vector = kolodkin_g_random_function::create_random_vector(1, -10000, 10000);
    std::vector<int> reference_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));
    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), false);
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_vector_with_two_elems) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = {50, 14};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(2, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
    ASSERT_EQ(global_out[0], 14);
    ASSERT_EQ(global_out[1], 50);
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_vector_with_three_elems) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = {50, 14, 5};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(3, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
    ASSERT_EQ(global_out[0], 5);
    ASSERT_EQ(global_out[1], 14);
    ASSERT_EQ(global_out[2], 50);
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_vector_with_negative_elems) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = {-50, -147, -5};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(3, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
    ASSERT_EQ(global_out[0], -147);
    ASSERT_EQ(global_out[1], -50);
    ASSERT_EQ(global_out[2], -5);
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_vector_with_repeated_elems) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(6, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = {50, 14, 5, 14, 5, 50};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(6, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_big_vector) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(1000, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = kolodkin_g_random_function::create_random_vector(1000, -10000, 10000);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(1000, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_big_vector_with_2n_size) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(1000, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = kolodkin_g_random_function::create_random_vector(8192, -10000, 10000);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(1000, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}

TEST(kolodkin_g_hoar_merge_sort_MPI, Test_big_vector_with_simple_value_size) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> vector;
  std::vector<int> global_out(1000, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  auto global_ptr = std::make_shared<std::vector<int>>(global_out);

  if (world.rank() == 0) {
    vector = kolodkin_g_random_function::create_random_vector(7541, -10000, 10000);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataMpi->inputs_count.emplace_back(vector.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_ptr.get()));
  }

  kolodkin_g_hoar_merge_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_out(1000, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());
    auto reference_ptr = std::make_shared<std::vector<int>>(reference_out);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_ptr.get()));

    kolodkin_g_hoar_merge_sort_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    global_out = *reinterpret_cast<std::vector<int> *>(taskDataMpi->outputs[0]);
    reference_out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
    for (unsigned i = 0; i < global_out.size(); i++) {
      ASSERT_EQ(global_out[i], reference_out[i]);
    }
  }
}
