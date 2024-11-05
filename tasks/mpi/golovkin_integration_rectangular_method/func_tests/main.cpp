// Golovkin Maksim
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

TEST(golovkin_integration_rectangular_method, test_constant_function) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 5.0;
  double epsilon = 0.1;

  if (world.rank() == 0) {
    // Инициализация данных на нулевом процессе
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallelTask(taskDataPar);

  // Выполнение параллельных задач на всех процессах
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создание и инициализация данных для последовательной задачи на нулевом процессе
    std::vector<double> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    // Выполнение последовательной задачи
    golovkin_integration_rectangular_method::MPIIntegralCalculator sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Сравнение результатов параллельной и последовательной интеграции
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-2);
  }
  exit(0);
}

TEST(golovkin_integration_rectangular_method, test_square_function) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 2.5;

  double epsilon = 0.1;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequentialTask(taskDataSeq);

    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-2);
  }
  exit(0);
}
TEST(golovkin_integration_rectangular_method, test_sine_function) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;

  double b = M_PI;
  double epsilon = 0.1;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequentialTask(taskDataSeq);

    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-2);
  }
  exit(0);
}

TEST(golovkin_integration_rectangular_method, test_exponential_function) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 2.5;  // Integrating e^x from 0 to 1
  double epsilon = 0.1;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequentialTask(taskDataSeq);

    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-2);
  }
  exit(0);
}

TEST(golovkin_integration_rectangular_method, test_polynomial_function) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 1.0;
  double b = 2.5;  // Integrating f(x) = x^3 from 1 to 3
  double epsilon = 0.1;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequentialTask(taskDataSeq);

    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-2);
  }
  exit(0);
}
