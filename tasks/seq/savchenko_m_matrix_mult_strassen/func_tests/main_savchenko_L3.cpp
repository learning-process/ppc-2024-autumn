#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <stdexcept>
#include <vector>

#include "seq/savchenko_m_matrix_mult_strassen/include/ops_seq_savchenko_L3.hpp"

namespace savchenko_m_matrix_mult_strassen_seq {
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

}  // namespace savchenko_m_matrix_mult_strassen_seq

TEST(savchenko_m_matrix_mult_strassen_seq, validation_inputs_count_zero) {
  // Create data
  const size_t size = 2;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_matrix_mult_strassen_seq, validation_inputs_count_two) {
  // Create data
  const size_t size = 2;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_matrix_mult_strassen_seq, validation_outputs_count_zero) {
  // Create data
  const size_t size = 2;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_matrix_mult_strassen_seq, validation_outputs_count_two) {
  // Create data
  const size_t size = 2;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_matrix_mult_strassen_seq, validation_matrix_size_zero) {
  // Create data
  const size_t size = 2;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(0);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(0);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(savchenko_m_matrix_mult_strassen_seq, simple_matrix_2x2) {
  const size_t size = 2;

  const double num_A = 2.0;
  const double num_B = 3.0;
  const double num_ref = size * num_A * num_B;

  std::vector<double> matrix_A(size * size, num_A);
  std::vector<double> matrix_B(size * size, num_B);

  std::vector<double> matrix_res(size * size, 0.0);
  std::vector<double> refference(size * size, num_ref);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  EXPECT_EQ(refference, matrix_res);
}

TEST(savchenko_m_matrix_mult_strassen_seq, simple_matrix_3x3) {
  const size_t size = 3;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  matrix_A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  matrix_B = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  std::vector<double> matrix_res(size * size, 0.0);
  std::vector<double> refference(size * size, 0.0);
  refference = {30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  EXPECT_EQ(refference, matrix_res);
}

TEST(savchenko_m_matrix_mult_strassen_seq, matrix_10x10) {
  // Create data
  const size_t size = 10;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<double> refference(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < size; ++k) {
        sum += matrix_A[i * size + k] * matrix_B[k * size + j];
      }
      refference[i * size + j] = sum;
    }
  }

  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
  }
}

TEST(savchenko_m_matrix_mult_strassen_seq, matrix_100x100) {
  // Create data
  const size_t size = 100;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<double> refference(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < size; ++k) {
        sum += matrix_A[i * size + k] * matrix_B[k * size + j];
      }
      refference[i * size + j] = sum;
    }
  }

  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
  }
}

TEST(savchenko_m_matrix_mult_strassen_seq, matrix_7x7) {
  // Create data
  const size_t size = 7;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<double> refference(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < size; ++k) {
        sum += matrix_A[i * size + k] * matrix_B[k * size + j];
      }
      refference[i * size + j] = sum;
    }
  }

  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
  }
}

TEST(savchenko_m_matrix_mult_strassen_seq, matrix_128x128) {
  // Create data
  const size_t size = 128;

  const double gen_min = -10.0;
  const double gen_max = 10.0;

  std::vector<double> matrix_A = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_B = savchenko_m_matrix_mult_strassen_seq::getRandomMatrix(size, gen_min, gen_max);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  taskDataSeq->inputs_count.emplace_back(size);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
  taskDataSeq->outputs_count.emplace_back(size);

  // Create Task
  savchenko_m_matrix_mult_strassen_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<double> refference(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < size; ++k) {
        sum += matrix_A[i * size + k] * matrix_B[k * size + j];
      }
      refference[i * size + j] = sum;
    }
  }

  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(refference[i], matrix_res[i], 1e-8);
  }
}
