#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> generate_random_vector(size_t size, size_t value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_random_matrix(size_t rows, size_t cols, size_t value) {
  std::vector<std::vector<int>> matrix(rows);
  for (size_t i = 0; i < rows; i++) {
    matrix[i] = generate_random_vector(cols, value);
  }
  return matrix;
}
TEST(sedova_o_max_of_vector_elements_mpi, Test1) { ASSERT_NO_THROW(generate_random_vector(10, 10)); }
TEST(sedova_o_max_of_vector_elements_mpi, Test2) { ASSERT_NO_THROW(generate_random_matrix(10, 10, 10)); }
TEST(sedova_o_max_of_vector_elements_mpi, Test3) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -30);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(1, 1, 20);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
    sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test4) {
  size_t rows = 1;
  size_t cols = 5;
  size_t value = 20;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -(int)value);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
    sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    testMpiTaskParallel.validation();
    EXPECT_EQ(testMpiTaskParallel.pre_processing(), true);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_10_10) {
  size_t rows = 10;
  size_t cols = 10;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_100_100) {
  size_t rows = 100;
  size_t cols = 100;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_1000_1000) {
  size_t rows = 1000;
  size_t cols = 1000;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_10_100) {
  size_t rows = 10;
  size_t cols = 100;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_100_10) {
  size_t rows = 100;
  size_t cols = 10;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_500_10) {
  size_t rows = 10;
  size_t cols = 500;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(sedova_o_max_of_vector_elements_mpi, Test_50_2) {
  size_t rows = 2;
  size_t cols = 50;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}


TEST(sedova_o_max_of_vector_elements_mpi, Test_10_2) {
  size_t rows = 2;
  size_t cols = 10;
  size_t value = 30;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, -((int)(value)));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = generate_random_matrix(rows, cols, value);
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, global_matrix[0][0]);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    // Create Task
    sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}