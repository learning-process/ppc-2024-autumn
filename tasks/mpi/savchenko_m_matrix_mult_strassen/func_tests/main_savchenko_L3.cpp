#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/savchenko_m_matrix_mult_strassen/include/ops_mpi_savchenko_L3.hpp"

namespace savchenko_m_matrix_mult_strassen_mpi {
std::vector<double> getRandomMatrix(size_t size, double min, double max) {
  if (size <= 0) {
    throw std::out_of_range("size must be greater than 0");
  }
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  // Forming a random matrix
  std::vector<double> matrix(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      matrix[i * size + j] = dist(gen);
    }
  }

  return matrix;
}

double getRandomDouble(double min, double max) {
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(gen);
}

}  // namespace savchenko_m_matrix_mult_strassen_mpi

TEST(savchenko_m_matrix_mult_strassen_mpi, validation_inputs_count_zero) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 2;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = std::vector<double>(size * size, 0.0);
    matrix_B = std::vector<double>(size * size, 0.0);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, validation_inputs_count_two) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 2;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = std::vector<double>(size * size, 0.0);
    matrix_B = std::vector<double>(size * size, 0.0);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, validation_outputs_count_zero) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 2;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = std::vector<double>(size * size, 0.0);
    matrix_B = std::vector<double>(size * size, 0.0);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));

    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, validation_outputs_count_two) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 2;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = std::vector<double>(size * size, 0.0);
    matrix_B = std::vector<double>(size * size, 0.0);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(size);

    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, validation_matrix_size_zero) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 2;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = std::vector<double>(size * size, 0.0);
    matrix_B = std::vector<double>(size * size, 0.0);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(0);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(0);

    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_5x5) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 5;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_10x10) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 10;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_15x15) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 15;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_20x20) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 20;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_8x8) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 8;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_16x16) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 16;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_32x32) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 32;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_64x64) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 64;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, matrixes_128x128) {
  // Create data
  boost::mpi::communicator world;

  const size_t size = 128;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;
  std::vector<double> refference;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const double gen_min = -10.0;
    const double gen_max = 10.0;

    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_res = std::vector<double>(size * size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);

    // Refference
    refference = std::vector<double>(size * size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(size);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refference.data()));
    taskDataSeq->outputs_count.emplace_back(size);
    // Create Task
    savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    ASSERT_TRUE(testMpiTaskSequential.pre_processing());
    ASSERT_TRUE(testMpiTaskSequential.run());
    ASSERT_TRUE(testMpiTaskSequential.post_processing());
  }
  // Create Task
  savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < size * size; i++) {
      EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
    }
  }
}
