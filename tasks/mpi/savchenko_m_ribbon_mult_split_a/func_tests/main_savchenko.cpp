#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/savchenko_m_ribbon_mult_split_a/include/ops_mpi_savchenko.hpp"

namespace savchenko_m_ribbon_mult_split_a_mpi {
std::vector<int> getRandomMatrix(size_t rows, size_t columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

int getRandomInt(int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int rand_int = min + gen() % (max - min + 1);
  return rand_int;
}
}  // namespace savchenko_m_ribbon_mult_split_a_mpi

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_zero_inputs) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_one_inputs) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_three_inputs) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_zero_outputs) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_two_outputs) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_zero_inputs_count) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_more_than_one_inputs_count) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_zero_outputs_count) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_more_than_one_outputs_count) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, validation_zero_matrix_size) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;
    matrix_A = std::vector<int>(res_size, 0);
    matrix_B = std::vector<int>(res_size, 0);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_5x5) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 5;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_10x10) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 10;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_15x15) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 15;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_50x50) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 50;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_100x100) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 100;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_128x128) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 128;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, matrixes_150x150) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;

  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 150;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }
  // Create Task
  savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    // Create data
    std::vector<int> refference(matrix_res.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    //// matrix_A
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// matrix_B
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(size);
    //// refference
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(refference.size());

    // Create Task
    savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());

    ASSERT_EQ(refference, matrix_res);
  }
}
