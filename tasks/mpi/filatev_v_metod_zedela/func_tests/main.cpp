// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

TEST(filatev_v_metod_zedela_mpi, test_size_3) {
  boost::mpi::communicator world;
  int size = 3;
  double alfa = 0.01;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

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

    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_10) {
  boost::mpi::communicator world;
  int size = 10;
  double alfa = 0.001;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

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

    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_100) {
  boost::mpi::communicator world;
  int size = 100;
  double alfa = 0.00001;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

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

    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_size_500) {
  boost::mpi::communicator world;
  int size = 500;
  double alfa = 0.00001;
  std::vector<double> answer;
  filatev_v_metod_zedela_mpi::TestClassForMetodZedela test;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    test.generatorMatrix(matrix, size);
    test.genetatirVectorB(matrix, vecB);

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

    ASSERT_EQ(test.rightAns(answer, alfa), true);
  }
}
