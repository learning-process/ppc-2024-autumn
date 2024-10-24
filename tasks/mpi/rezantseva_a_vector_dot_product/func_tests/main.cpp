// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/rezantseva_a_vector_dot_product/include/ops_mpi.hpp"

TEST(rezantseva_a_vector_dot_product_mpi, can_scalar_multiply_vec_size_120) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int count_size_vector = 120;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(rezantseva_a_vector_dot_product_mpi, can_scalar_multiply_vec_size_300) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int count_size_vector = 120;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    rezantseva_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(rezantseva_a_vector_dot_product_mpi, check_vectors_not_equal) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int count_size_vector = 120;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector + 5);
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  // Create Task
  rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(rezantseva_a_vector_dot_product_mpi, check_vectors_equal_true) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const int count_size_vector = 120;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_mpi::createRandomVector(count_size_vector);
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  // Create Task
  rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
}

TEST(rezantseva_a_vector_dot_product_mpi, check_mpi_vectorDotProduct_right) {
  // Create data
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, rezantseva_a_vector_dot_product_mpi::vectorDotProduct(v1, v2));
}

TEST(rezantseva_a_vector_dot_product_mpi, can_throw_exp_mpi_vectorDotProduct) {
  // Create data
  std::vector<int> v1 = {1, 2, 5, 7};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_ANY_THROW(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(v1, v2));
}

TEST(rezantseva_a_vector_dot_product_mpi, check_mpi_run_right) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = {1, 2, 5, 6};
  std::vector<int> v2 = {4, 7, 8, 9};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  rezantseva_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(rezantseva_a_vector_dot_product_mpi::vectorDotProduct(v1, v2), res[0]);
}
