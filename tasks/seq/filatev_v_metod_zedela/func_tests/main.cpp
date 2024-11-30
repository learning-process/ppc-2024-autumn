// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <vector>

#include "seq/filatev_v_metod_zedela/include/ops_seq.hpp"

using namespace filatev_v_metod_zedela_seq;

TEST(filatev_v_metod_zedela_seq, test_3) {
  int size = 3;
  double alfa = 0.01;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;

  filatev_v_metod_zedela_seq::TestClassForMetodZedela test;
  test.generatorMatrix(matrix, size);
  test.genetatirVectorB(matrix, vecB);

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

  ASSERT_EQ(test.rightAns(answer, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_5) {
  int size = 5;
  double alfa = 0.0001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;

  filatev_v_metod_zedela_seq::TestClassForMetodZedela test;
  test.generatorMatrix(matrix, size);
  test.genetatirVectorB(matrix, vecB);

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

  ASSERT_EQ(test.rightAns(answer, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_10) {
  int size = 10;
  double alfa = 0.00001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;

  filatev_v_metod_zedela_seq::TestClassForMetodZedela test;
  test.generatorMatrix(matrix, size);
  test.genetatirVectorB(matrix, vecB);

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

  ASSERT_EQ(test.rightAns(answer, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_6) {
  int size = 3;
  double alfa = 0.0001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;

  filatev_v_metod_zedela_seq::TestClassForMetodZedela test;
  test.generatorMatrix(matrix, size);
  test.genetatirVectorB(matrix, vecB);

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

  ASSERT_EQ(test.rightAns(answer, alfa), true);
}

TEST(filatev_v_metod_zedela_seq, test_error1) {
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

TEST(filatev_v_metod_zedela_seq, test_error2) {
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

TEST(filatev_v_metod_zedela_seq, test_error3) {
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

TEST(filatev_v_metod_zedela_seq, test_maxi_rz) {
  int size = 500;
  double alfa = 0.0001;
  std::vector<int> matrix(size * size);
  std::vector<int> vecB(size);
  std::vector<double> answer;

  filatev_v_metod_zedela_seq::TestClassForMetodZedela test;
  test.generatorMatrix(matrix, size);
  test.genetatirVectorB(matrix, vecB);

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

  ASSERT_EQ(test.rightAns(answer, alfa), true);
}