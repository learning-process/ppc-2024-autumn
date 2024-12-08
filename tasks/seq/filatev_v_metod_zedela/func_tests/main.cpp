// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <vector>

#include "seq/filatev_v_metod_zedela/include/ops_seq.hpp"

namespace filatev_v_metod_zedela_seq {

int generatorVector(std::vector<int> &vec) {
  int sum = 0;
  for (long unsigned int i = 0; i < vec.size(); ++i) {
    vec[i] = rand() % 100 - 50;
    sum += abs(vec[i]);
  }
  return sum;
}

void generatorMatrix(std::vector<int> &matrix, int size) {
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) {
      matrix[i * size + j] = rand() % 100 - 51;
      sum += abs(matrix[i * size + j]);
    }
    matrix[i * size + i] = sum + rand() % 100;
  }
}

std::vector<int> genetatirVectorB(std::vector<int> &matrix, std::vector<int> &vecB) {
  int size = vecB.size();
  std::vector<int> ans(size);
  generatorVector(ans);
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) {
      sum += matrix[j + i * size] * ans[j];
    }
    vecB[i] = sum;
  }
  return ans;
}

bool rightAns(std::vector<double> &ans, std::vector<int> &resh, double alfa) {
  double max_r = 0;
  for (long unsigned int i = 0; i < ans.size(); ++i) {
    double temp = abs(ans[i] - resh[i]);
    max_r = std::max(max_r, temp);
  }
  return max_r < alfa;
}

}  // namespace filatev_v_metod_zedela_seq

TEST(filatev_v_metod_zedela_seq, test_size_3) {
  int size = 3;
  double alfa = 0.01;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_size_5) {
  int size = 5;
  double alfa = 0.0001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_size_10) {
  int size = 10;
  double alfa = 0.00001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_error_rank) {
  int size = 3;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);

  matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vecB = {20, 11, 16};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);

  ASSERT_EQ(metodZedela.validation(), false);
}

TEST(filatev_v_metod_zedela_seq, test_error_determenant) {
  int size = 2;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);

  matrix = {1, 2, 2, 4};
  vecB = {3, 6};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);

  ASSERT_EQ(metodZedela.validation(), false);
}

TEST(filatev_v_metod_zedela_seq, test_error_diagonal) {
  int size = 3;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);

  matrix = {2, 16, 3, 11, 5, 10, 7, 8, 25};
  vecB = {20, 11, 16};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);

  ASSERT_EQ(metodZedela.validation(), false);
}

TEST(filatev_v_metod_zedela_seq, test_error_different_size) {
  int size = 3;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size + 1);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);

  ASSERT_EQ(metodZedela.validation(), false);
}

TEST(filatev_v_metod_zedela_seq, test_maxi_rz) {
  int size = 500;
  double alfa = 0.0001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;
  std::vector<int> resh;

  filatev_v_metod_zedela_seq::generatorMatrix(matrix, size);
  resh = filatev_v_metod_zedela_seq::genetatirVectorB(matrix, vecB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs_count.emplace_back(size);

  filatev_v_metod_zedela_seq::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
  answer.insert(answer.end(), temp, temp + size);

  ASSERT_EQ(filatev_v_metod_zedela_seq::rightAns(answer, resh, alfa), true);
}