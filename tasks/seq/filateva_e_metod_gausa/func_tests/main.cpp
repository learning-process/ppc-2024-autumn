// Filateva Elizaveta Metod Gausa
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "seq/filateva_e_metod_gausa/include/ops_seq.hpp"

#define alfa std::numeric_limits<double>::epsilon() * 10000

std::vector<double> gereratorSLU(std::vector<double> &matrix, std::vector<double> &vecB) {
  int min_z = -100;
  int max_z = 100;
  int size = vecB.size();
  std::vector<double> resh(size);
  for (int i = 0; i < size; i++) {
    resh[i] = rand() % (max_z - min_z + 1) + min_z;
  }
  for (int i = 0; i < size; i++) {
    double sum = 0;
    double sumB = 0;
    for (int j = 0; j < size; j++) {
      matrix[i * size + j] = rand() % (max_z - min_z + 1) + min_z;
      sum += abs(matrix[i * size + j]);
    }
    matrix[i * size + i] = sum;
    for (int j = 0; j < size; j++) {
      sumB += matrix[i * size + j] * resh[j];
    }
    vecB[i] = sumB;
  }
  return resh;
}

TEST(filateva_e_metod_gausa_seq, test_size_3) {
  int size = 3;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer(size);
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_size_10) {
  int size = 10;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer(size);
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_size_100) {
  int size = 100;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer(size);
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_size_200) {
  int size = 200;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer(size);
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_size_different) {
  int size = 10;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size + 1);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  EXPECT_FALSE(metodGausa.validation());
}

TEST(filateva_e_metod_gausa_seq, test_size_0) {
  int size = 0;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  EXPECT_FALSE(metodGausa.validation());
}

TEST(filateva_e_metod_gausa_seq, test_error_rank) {
  int size = 3;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);

  matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vecB = {20, 11, 16};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size + 1);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  EXPECT_FALSE(metodGausa.validation());
}

TEST(filateva_e_metod_gausa_seq, test_error_determenant) {
  int size = 2;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);

  matrix = {1, 2, 2, 4};
  vecB = {3, 6};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size + 1);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  EXPECT_FALSE(metodGausa.validation());
}

TEST(filateva_e_metod_gausa_seq, test_diagonal_0) {
  int size = 3;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer(size);
  std::vector<double> tResh;

  matrix = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  vecB = {8, 8, 8};
  tResh = {4, 4, 4};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_diagonal_0_2) {
  int size = 3;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer(size);
  std::vector<double> tResh;

  matrix = {0, 1, 1, 1, 1, 1, 0, 1, 0};
  vecB = {2, 3, 1};
  tResh = {1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}

TEST(filateva_e_metod_gausa_seq, test_size_3_diagonal_0) {
  int size = 3;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer(size);
  std::vector<double> tResh;

  matrix = {1, 3, 2, 2, 6, 1, 7, 0, 9};
  vecB = {10, 12, 11};
  tResh = {(double)-13 / 7, (double)137 / 63, (double)8 / 3};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_TRUE(metodGausa.validation());
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  EXPECT_EQ(answer.size(), tResh.size());
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(tResh[i], answer[i], alfa);
  }
}