#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <random>
#include <vector>

#include "mpi/malyshev_a_sum_rows_matrix/include/ops_mpi.hpp"

namespace malyshev_a_sum_rows_matrix_test_function {

std::vector<std::vector<int32_t>> getRandomData(uint32_t rows, uint32_t cols) {
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

}  // namespace malyshev_a_sum_rows_matrix_test_function

TEST(malyshev_a_sum_rows_matrix_mpi, rectangular_matrix_stretched_horizontally_7x17) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqSum(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_a_sum_rows_matrix_mpi::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomData) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
    taskDataSeq->outputs_count.push_back(seqSum.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiSum.size(); i++) {
      ASSERT_EQ(seqSum[i], mpiSum[i]);
    }
  }
}

TEST(malyshev_a_sum_rows_matrix_mpi, rectangular_matrix_stretched_verticaly_100x75) {
  uint32_t rows = 100;
  uint32_t cols = 75;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqSum(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_a_sum_rows_matrix_mpi::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomData) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
    taskDataSeq->outputs_count.push_back(seqSum.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiSum.size(); i++) {
      ASSERT_EQ(seqSum[i], mpiSum[i]);
    }
  }
}

TEST(malyshev_a_sum_rows_matrix_mpi, squere_matrix_100x100) {
  uint32_t rows = 100;
  uint32_t cols = 100;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqSum(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_a_sum_rows_matrix_mpi::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomData) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
    taskDataSeq->outputs_count.push_back(seqSum.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiSum.size(); i++) {
      ASSERT_EQ(seqSum[i], mpiSum[i]);
    }
  }
}

TEST(malyshev_a_sum_rows_matrix_mpi, matrix_1x1) {
  uint32_t rows = 1;
  uint32_t cols = 1;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    std::vector<int32_t> seqSum(rows);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    malyshev_a_sum_rows_matrix_mpi::TestTaskSequential taskSeq(taskDataSeq);

    for (auto &row : randomData) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataSeq->inputs_count.push_back(rows);
    taskDataSeq->inputs_count.push_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
    taskDataSeq->outputs_count.push_back(seqSum.size());

    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    for (uint32_t i = 0; i < mpiSum.size(); i++) {
      ASSERT_EQ(seqSum[i], mpiSum[i]);
    }
  }
}

TEST(malyshev_a_sum_rows_matrix_mpi, test_validation) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(0);

    ASSERT_FALSE(taskMPI.validation());
  }
}