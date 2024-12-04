// Filateva Elizaveta Metod Gausa
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <limits>
#include <random>
#include <vector>

#include "mpi/filateva_e_metod_gausa/include/ops_mpi.hpp"

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

bool check(std::vector<double> &resh, std::vector<double> &tResh, double alfa) {
  for (unsigned long i = 0; i < tResh.size(); i++) {
    if (abs(resh[i] - tResh[i]) > alfa) {
      return false;
    }
  }
  return true;
}

TEST(filateva_e_metod_gausa_mpi, test_size_3) {
  boost::mpi::communicator world;
  int size = 3;
  double alfa = std::numeric_limits<double>::epsilon() * 10000;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(check(answer, tResh, alfa), true);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_10) {
  boost::mpi::communicator world;
  int size = 10;
  double alfa = std::numeric_limits<double>::epsilon() * 10000;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(check(answer, tResh, alfa), true);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_30) {
  boost::mpi::communicator world;
  int size = 30;
  double alfa = std::numeric_limits<double>::epsilon() * 10000;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(check(answer, tResh, alfa), true);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_100) {
  boost::mpi::communicator world;
  int size = 100;
  double alfa = std::numeric_limits<double>::epsilon() * 10000;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(check(answer, tResh, alfa), true);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_200) {
  boost::mpi::communicator world;
  int size = 200;
  double alfa = std::numeric_limits<double>::epsilon() * 10000;
  std::vector<double> matrix;
  std::vector<double> vecB;
  std::vector<double> answer;
  std::vector<double> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);
    tResh = gereratorSLU(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  ASSERT_EQ(metodGausa.validation(), true);
  metodGausa.pre_processing();
  metodGausa.run();
  metodGausa.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(check(answer, tResh, alfa), true);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_different) {
  boost::mpi::communicator world;
  int size = 5;
  std::vector<double> matrix;
  std::vector<double> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size + 1);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  if (world.rank() == 0) {
    ASSERT_EQ(metodGausa.validation(), false);
  }
}

TEST(filateva_e_metod_gausa_mpi, test_size_0) {
  boost::mpi::communicator world;
  int size = 0;
  std::vector<double> matrix;
  std::vector<double> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size * size);
    vecB.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_metod_gausa_mpi::MetodGausa metodGausa(taskData);

  if (world.rank() == 0) {
    ASSERT_EQ(metodGausa.validation(), false);
  }
}
