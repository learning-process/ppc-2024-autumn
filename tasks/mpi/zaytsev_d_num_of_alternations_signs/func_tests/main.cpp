// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/zaytsev_d_num_of_alternations_signs/include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100 - 50;
  }
  return vec;
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, AllPositive) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, AllNegative) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, AlternatingPositiveNegative) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, WithZeros) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 0, -1, 0, 1, 0, -1, 0};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, BigVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, SmallVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, EmptyVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, WithRandomVector) {
  boost::mpi::communicator world;
  int vector_size = 100;
  std::vector<int> random_vector = getRandomVector(vector_size);
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataSeq->inputs_count.emplace_back(random_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_count[0], reference_count[0]);
  }
}
