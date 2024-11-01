// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zinoviev_a_sum_cols_matrix/include/ops_mpi.hpp"

TEST(zinoviev_a_sum_cols_matrix, test_task_run_small_matrix) {
  boost::mpi::communicator world;
  int cols = 10;
  int rows = 10;

  // Создание данных
  std::vector<int> matrix(cols * rows, 0);
  matrix[3] = 1;  // Устанавливаем случайное значение
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[3] = 1;

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}

TEST(zinoviev_a_sum_cols_matrix, test_task_run_zero_matrix) {
  boost::mpi::communicator world;
  int cols = 100;
  int rows = 100;

  // Создание нулевой матрицы
  std::vector<int> matrix(cols * rows, 0);
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}

TEST(zinoviev_a_sum_cols_matrix, test_pipeline_run_identical_rows) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 1000;

  // Создание матрицы с идентичными строками
  std::vector<int> matrix(cols * rows, 1);  // Каждое значение равно 1
  std::vector<int> expres(cols, 1000);      // Сумма каждого столбца
  std::vector<int> ans(cols, 1000);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}

TEST(zinoviev_a_sum_cols_matrix, test_run_for_1_col) {
  boost::mpi::communicator world;

  int cols = 1;
  int rows = 100;

  // Создание данных
  std::vector<int> matrix(cols * rows, 1);  // Все значения равны 1
  std::vector<int> expres(cols, 100);       // Ожидаемая сумма равна 100
  std::vector<int> ans(cols, 100);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}

TEST(zinoviev_a_sum_cols_matrix, test_run_for_1_row) {
  boost::mpi::communicator world;

  int cols = 10;
  int rows = 1;

  // Создание данных
  std::vector<int> matrix(cols * rows, 0);
  matrix[2] = 7;  // Установим значение в столбце 2
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[2] = 7;  // Ожидаемая сумма в третьем столбце

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}
