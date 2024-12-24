// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "mpi/fyodorov_m_trapezoidal_method_mpi/include/ops_mpi.hpp"

TEST(fyodorov_m_trapezoidal_method_mpi, Test_Integration) {
  boost::mpi::communicator world;

  // Создаем данные для теста
  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) {
    return 1.0;  // Простая функция для интегрирования: f(x) = 1
  };

  std::vector<double> lower_bounds = {0.0};  // Нижняя граница
  std::vector<double> upper_bounds = {1.0};  // Верхняя граница
  std::vector<int> intervals = {10};         // Количество интервалов

  std::vector<double> global_result(1, 0.0);  // Глобальный результат

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {1, 1, 1};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  // Создаем и запускаем параллельную задачу
  fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, func);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<double> reference_result(1, 0.0);  // Результат последовательной версии

    // Создаем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataSeq->inputs_count = {1, 1, 1, 1};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count = {1};

    // Создаем и запускаем последовательную задачу
    fyodorov_m_trapezoidal_method_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    // Сравниваем результаты
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-6);
  }
}

TEST(fyodorov_m_trapezoidal_method_mpi, Test_Integration_2) {
  boost::mpi::communicator world;

  // Создаем данные для теста
  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) {
    return 1.0;  // Простая функция для интегрирования: f(x) = 1
  };

  std::vector<double> lower_bounds = {0.0, 0.0};  // Нижняя граница
  std::vector<double> upper_bounds = {1.0, 1.0};  // Верхняя граница
  std::vector<int> intervals = {10, 10};          // Количество интервалов

  std::vector<double> global_result(1, 0.0);  // Глобальный результат

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {2, 2, 2};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  // Создаем и запускаем параллельную задачу
  fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, func);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<double> reference_result(1, 0.0);  // Результат последовательной версии

    // Создаем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataSeq->inputs_count = {1, 2, 2, 2};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count = {1};

    // Создаем и запускаем последовательную задачу
    fyodorov_m_trapezoidal_method_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    // Сравниваем результаты
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-6);
  }
}

TEST(fyodorov_m_trapezoidal_method_mpi, Test_Integration_ConstantFunction) {
  boost::mpi::communicator world;

  // Создаем данные для теста
  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) {
    return 2.0;  // Константная функция для интегрирования: f(x) = 2
  };

  std::vector<double> lower_bounds = {0.0};  // Нижняя граница
  std::vector<double> upper_bounds = {1.0};  // Верхняя граница
  std::vector<int> intervals = {10};         // Количество интервалов

  std::vector<double> global_result(1, 0.0);  // Глобальный результат

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {1, 1, 1};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  // Создаем и запускаем параллельную задачу
  fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, func);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<double> reference_result(1, 0.0);  // Результат последовательной версии

    // Создаем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataSeq->inputs_count = {1, 1, 1, 1};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count = {1};

    // Создаем и запускаем последовательную задачу
    fyodorov_m_trapezoidal_method_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    // Сравниваем результаты
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-6);
  }
}

TEST(fyodorov_m_trapezoidal_method_mpi, Test_Integration_LinearFunction) {
  boost::mpi::communicator world;

  // Создаем данные для теста
  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) {
    return point[0];  // Линейная функция для интегрирования: f(x) = x
  };

  std::vector<double> lower_bounds = {0.0};  // Нижняя граница
  std::vector<double> upper_bounds = {1.0};  // Верхняя граница
  std::vector<int> intervals = {10};         // Количество интервалов

  std::vector<double> global_result(1, 0.0);  // Глобальный результат

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {1, 1, 1};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  // Создаем и запускаем параллельную задачу
  fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, func);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<double> reference_result(1, 0.0);  // Результат последовательной версии

    // Создаем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataSeq->inputs_count = {1, 1, 1, 1};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count = {1};

    // Создаем и запускаем последовательную задачу
    fyodorov_m_trapezoidal_method_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    // Сравниваем результаты
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}
