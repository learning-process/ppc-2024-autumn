#include <gtest/gtest.h>

#include "mpi/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderMPI.hpp"

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_validation_empty_vector) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(4, 12);
  std::vector<int> inputVector = {};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_validation_empty_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = {};
  std::vector<int> inputVector = {1, 2, 3};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(0);
    taskDataParallel->inputs_count.emplace_back(0);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_validation_small_vector) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(4, 12);
  std::vector<int> inputVector = {1, 2, 3};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_validation_big_vector) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(4, 12);
  std::vector<int> inputVector = {1, 2, 3, 4, 5};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_4x12_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(4, 12);
  std::vector<int> inputVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(4);
  std::vector<int> resultVector(12, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceResultVector(12, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataSequential->inputs_count.emplace_back(4);
    taskDataSequential->inputs_count.emplace_back(12);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataSequential->inputs_count.emplace_back(inputVector.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceResultVector.data()));
    taskDataSequential->outputs_count.emplace_back(referenceResultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(resultVector, referenceResultVector);
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_120x120_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(120, 120);
  std::vector<int> inputVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(120);
  std::vector<int> resultVector(120, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(120);
    taskDataParallel->inputs_count.emplace_back(120);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceResultVector(120, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataSequential->inputs_count.emplace_back(120);
    taskDataSequential->inputs_count.emplace_back(120);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataSequential->inputs_count.emplace_back(inputVector.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceResultVector.data()));
    taskDataSequential->outputs_count.emplace_back(referenceResultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(resultVector, referenceResultVector);
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_12x900_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(12, 900);
  std::vector<int> inputVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(12);
  std::vector<int> resultVector(900, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs_count.emplace_back(900);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceResultVector(900, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataSequential->inputs_count.emplace_back(12);
    taskDataSequential->inputs_count.emplace_back(900);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataSequential->inputs_count.emplace_back(inputVector.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceResultVector.data()));
    taskDataSequential->outputs_count.emplace_back(referenceResultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(resultVector, referenceResultVector);
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_900x12_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(900, 12);
  std::vector<int> inputVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(900);
  std::vector<int> resultVector(12, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(900);
    taskDataParallel->inputs_count.emplace_back(12);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceResultVector(12, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataSequential->inputs_count.emplace_back(900);
    taskDataSequential->inputs_count.emplace_back(12);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataSequential->inputs_count.emplace_back(inputVector.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceResultVector.data()));
    taskDataSequential->outputs_count.emplace_back(referenceResultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(resultVector, referenceResultVector);
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_2560x1440_matrix) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(2560, 1440);
  std::vector<int> inputVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(2560);
  std::vector<int> resultVector(1440, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(2560);
    taskDataParallel->inputs_count.emplace_back(1440);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceResultVector(1440, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataSequential->inputs_count.emplace_back(2560);
    taskDataSequential->inputs_count.emplace_back(1440);
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataSequential->inputs_count.emplace_back(inputVector.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceResultVector.data()));
    taskDataSequential->outputs_count.emplace_back(referenceResultVector.size());

    lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(resultVector, referenceResultVector);
  }
}