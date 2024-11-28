// Filateva Elizaveta Metod Gausa
#include <gtest/gtest.h>

#include <vector>

#include "seq/filateva_e_metod_gausa/include/ops_seq.hpp"

std::vector<double> gereratorSLU(std::vector<double> &matrix, std::vector<double> &vecB) {
  int min_z = -5;
  int max_z = 5;
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

bool check(std::vector<double> &resh, std::vector<double> &tResh, double alfa) {
  for (long unsigned int i = 0; i < tResh.size(); i++) {
    if (abs(resh[i] - tResh[i]) > alfa) {
      return false;
    }
  }
  return true;
}

TEST(filateva_e_metod_gausa_seq, test_1) {
  int size = 3;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(check(answer, tResh, alfa), true);
}

TEST(filateva_e_metod_gausa_seq, test_2) {
  int size = 10;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(check(answer, tResh, alfa), true);
}

TEST(filateva_e_metod_gausa_seq, test_3) {
  int size = 100;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(check(answer, tResh, alfa), true);
}

TEST(filateva_e_metod_gausa_seq, test_4) {
  int size = 200;
  double alfa = 0.000000001;
  std::vector<double> matrix(size * size);
  std::vector<double> vecB(size);
  std::vector<double> answer;
  std::vector<double> tResh;

  tResh = gereratorSLU(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filateva_e_metod_gausa_seq::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(check(answer, tResh, alfa), true);
}