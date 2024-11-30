// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

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
    std::vector<int> temp(size);
    int sum = generatorVector(temp);
    temp[i] = sum + rand() % 100;
    matrix.insert(matrix.begin() + i * size, temp.begin(), temp.end());
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

TEST(filatev_v_metod_zedela_mpi, test_size_3) {
  boost::mpi::communicator world;
  int size = 3;
  double alfa = 0.01;
  std::vector<double> answer;
  std::vector<int> resh;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    generatorMatrix(matrix, size);
    resh = genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(rightAns(answer, resh, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_10) {
  boost::mpi::communicator world;
  int size = 10;
  double alfa = 0.001;
  std::vector<double> answer;
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::vector<int> resh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    generatorMatrix(matrix, size);
    resh = genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(rightAns(answer, resh, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_100) {
  boost::mpi::communicator world;
  int size = 100;
  double alfa = 0.00001;
  std::vector<double> answer;
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::vector<int> resh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    generatorMatrix(matrix, size);
    resh = genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(rightAns(answer, resh, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_500) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.00001;
  std::vector<double> answer;
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::vector<int> resh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    generatorMatrix(matrix, size);
    resh = genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(rightAns(answer, resh, alfa), true);
  }
}
