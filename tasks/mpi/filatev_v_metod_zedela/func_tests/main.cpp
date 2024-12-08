// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

namespace filatev_v_metod_zedela_mpi {

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
      matrix[i * size + j] = rand() % 100 - 50;
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

}  // namespace filatev_v_metod_zedela_mpi

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

    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
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

    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
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

    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
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

    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (world.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_error_rank) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<int> matrix;
  std::vector<int> vecB;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vecB = {20, 11, 16};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);

  if (world.rank() == 0) {
    ASSERT_EQ(metodZedela.validation(), false);
  } else {
    ASSERT_EQ(metodZedela.validation(), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_error_determenant) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 2, 4};
    vecB = {3, 6};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);

  if (world.rank() == 0) {
    ASSERT_EQ(metodZedela.validation(), false);
  } else {
    ASSERT_EQ(metodZedela.validation(), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_error_different_size) {
  boost::mpi::communicator world;
  int size = 2;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {1, 2, 2, 4};
    vecB = {3, 6};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size + 1);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);

  if (world.rank() == 0) {
    ASSERT_EQ(metodZedela.validation(), false);
  } else {
    ASSERT_EQ(metodZedela.validation(), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_error_diagonal) {
  boost::mpi::communicator world;
  int size = 3;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {2, 16, 3, 11, 5, 10, 7, 8, 25};
    vecB = {20, 11, 16};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, world);

  if (world.rank() == 0) {
    ASSERT_EQ(metodZedela.validation(), false);
  } else {
    ASSERT_EQ(metodZedela.validation(), true);
  }
}

TEST(filatev_v_metod_zedela_mpi, test_codecov_1_procces) {
  boost::mpi::communicator world;
  int size = 3;
  double alfa = 0.01;
  std::vector<double> answer;
  std::vector<int> resh;
  std::vector<int> matrix;
  std::vector<int> vecB;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  int color = static_cast<int>(world.rank() == 0);
  boost::mpi::communicator new_comm = world.split(color);

  if (new_comm.rank() == 0) {
    matrix.resize(size * size, 0);
    vecB.resize(size, 0);

    filatev_v_metod_zedela_mpi::generatorMatrix(matrix, size);
    resh = filatev_v_metod_zedela_mpi::genetatirVectorB(matrix, vecB);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vecB.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filatev_v_metod_zedela_mpi::MetodZedela metodZedela(taskData, new_comm);
  metodZedela.setAlfa(alfa);

  ASSERT_EQ(metodZedela.validation(), true);
  metodZedela.pre_processing();
  metodZedela.run();
  metodZedela.post_processing();

  if (new_comm.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(taskData->outputs[0]);
    answer.insert(answer.end(), temp, temp + size);

    ASSERT_EQ(filatev_v_metod_zedela_mpi::rightAns(answer, resh, alfa), true);
  }
}