#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_lent_horizontal/include/ops_mpi.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = -200 + gen() % (300 + 200 + 1);
    }
  }

  return data;
}

std::vector<int32_t> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = -200 + gen() % (300 + 200 + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_vertical_stretched_matrix_100x75) {
  uint32_t rows = 100;
  uint32_t cols = 75;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_horizontal_stretched_matrix_7x17) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_square_matrix_100x100) {
  uint32_t rows = 100;
  uint32_t cols = 100;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_single_element_matrix_1x1) {
  uint32_t rows = 1;
  uint32_t cols = 1;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqResult(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomMatrix) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
    taskDataSeq->outputs_count.push_back(seqResult.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiResult.size(); i++) {
      ASSERT_EQ(seqResult[i], mpiResult[i]);
    }
  }
}

TEST(malyshev_lent_horizontal, test_validation_failure) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(0);

    ASSERT_FALSE(taskMPI.validation());
  }
}