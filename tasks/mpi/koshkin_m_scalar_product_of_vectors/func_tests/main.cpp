// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/koshkin_m_scalar_product_of_vectors/include/ops_mpi.hpp"
static int offset = 0;

std::vector<int> createRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

TEST(koshkin_m_scalar_product_of_vectors, check_vec_equal) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100;
    std::vector<int> vec_1 = createRandomVector(count_size_vector);
    std::vector<int> vec_2 = createRandomVector(count_size_vector);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, check_vec_no_equal) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100;
    std::vector<int> vec_1 = createRandomVector(count_size_vector);
    std::vector<int> vec_2 = createRandomVector(count_size_vector + 10);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, multiply_vec_size_100) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 100;
    std::vector<int> vec_1 = createRandomVector(count_size_vector);
    std::vector<int> vec_2 = createRandomVector(count_size_vector);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    koshkin_m_scalar_product_of_vectors::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, multiply_vec_size_300) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 300;
    std::vector<int> vec_1 = createRandomVector(count_size_vector);
    std::vector<int> vec_2 = createRandomVector(count_size_vector);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    koshkin_m_scalar_product_of_vectors::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, multiply_vec_size_600) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 600;
    std::vector<int> vec_1 = createRandomVector(count_size_vector);
    std::vector<int> vec_2 = createRandomVector(count_size_vector);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    koshkin_m_scalar_product_of_vectors::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_vectorDotProduct_binary) {
  std::vector<int> vec_1 = {5, 2};
  std::vector<int> vec_2 = {6, 10};
  ASSERT_EQ(50, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_vectorDotProduct_ternary) {
  std::vector<int> vec_1 = {5, 2, 10};
  std::vector<int> vec_2 = {6, 10, 1};
  ASSERT_EQ(60, koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2));
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_run_size_4) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> vec_1 = {1, 2, 5, 6};
  std::vector<int> vec_2 = {4, 7, 8, 9};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_run_size_7) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> vec_1 = {1, 2, 5, 6, 9, 13, 5};
  std::vector<int> vec_2 = {4, 7, 8, 9, 8, 4, 17};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_run_size_rand) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> vec_1 = {1, 2, 5, 6, 9, 13, 5};
  std::vector<int> vec_2 = {4, 7, 8, 9, 8, 4, 17};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2), res[0]);
  }
}

TEST(koshkin_m_scalar_product_of_vectors, check_mpi_run_random_size) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  int random_size = 1 + std::rand() % 100;

  if (world.rank() == 0) {
    std::vector<int> vec_1 = createRandomVector(random_size);
    std::vector<int> vec_2 = createRandomVector(random_size);

    global_vec = {vec_1, vec_2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  koshkin_m_scalar_product_of_vectors::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(koshkin_m_scalar_product_of_vectors::calculateDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}
