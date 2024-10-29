#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kudryashova_i_vector_dot_product/include/vectorDotProductMPI.hpp"

TEST(kudryashova_i_vector_dot_product_mpi, mpi_vectorDotProduct) {
  std::vector<int> vector1 = {8, 7, 6};
  std::vector<int> vector2 = {3, 2, 1};
  ASSERT_EQ(44, kudryashova_i_vector_dot_product_mpi::vectorDotProduct(vector1, vector2));
}

TEST(kudryashova_i_vector_dot_product_mpi, scalar_multiply_vector_120) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 120;
    std::vector<int> vector1 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    std::vector<int> vector2 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    global_vector = {vector1, vector2};
    for (size_t i = 0; i < global_vector.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vector[0].size());
    taskDataPar->inputs_count.emplace_back(global_vector[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vector.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vector[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vector[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    // Create Task
    kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(kudryashova_i_vector_dot_product_mpi::vectorDotProduct(global_vector[0], global_vector[1]), result[0]);
    ASSERT_EQ(reference[0], result[0]);
  }
}

TEST(kudryashova_i_vector_dot_product_mpi, scalar_multiply_vector_300) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 300;
    std::vector<int> vector1 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    std::vector<int> vector2 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    global_vector = {vector1, vector2};
    for (size_t i = 0; i < global_vector.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vector[0].size());
    taskDataPar->inputs_count.emplace_back(global_vector[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vector.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
    }
    taskDataSeq->inputs_count.emplace_back(global_vector[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vector[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    // Create Task
    kudryashova_i_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(kudryashova_i_vector_dot_product_mpi::vectorDotProduct(global_vector[0], global_vector[1]), result[0]);
    ASSERT_EQ(reference[0], result[0]);
  }
}

TEST(kudryashova_i_vector_dot_product_mpi, check_vectors_equal) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vector;
  std::vector<int32_t> result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 100;
    std::vector<int> vector1 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    std::vector<int> vector2 = kudryashova_i_vector_dot_product_mpi::getRandomVector(count_size_vector);
    global_vector = {vector1, vector2};
    for (size_t i = 0; i < global_vector.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(global_vector[0].size());
    taskDataPar->inputs_count.emplace_back(global_vector[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }
  kudryashova_i_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
}