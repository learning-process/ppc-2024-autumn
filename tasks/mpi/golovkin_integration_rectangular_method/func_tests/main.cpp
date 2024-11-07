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
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 10.0;
  int partition_count = -1000;  // Устанавливаем некорректное значение, чтобы вызвать ошибку

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return 5.0; });

  // Пробуем запустить и ожидаем, что validation() вызовет MPI_Abort
  if (comm_world.rank() == 0) {
    ASSERT_EQ(parallel_task.validation(), false);  // Валидация должна провалиться
  } else {
    parallel_task.validation();  // Ожидается синхронизация с rank 0, результат broadcast
  }

  // Если валидация прошла (что не должно быть в этом тесте), продолжаем выполнение
  if (parallel_task.validation()) {
    std::cout << "Rank " << comm_world.rank() << " started pre_processing.\n";
    parallel_task.pre_processing();
    std::cout << "Rank " << comm_world.rank() << " finished pre_processing.\n";
    std::cout << "Rank " << comm_world.rank() << " started run.\n";
    parallel_task.run();
    std::cout << "Rank " << comm_world.rank() << " finished run.\n";
    std::cout << "Rank " << comm_world.rank() << " started post_processing.\n";
    parallel_task.post_processing();
    std::cout << "Rank " << comm_world.rank() << " finished post_processing.\n";

    if (comm_world.rank() == 0) {
      std::vector<double> expected_result(1, 0);
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
      taskDataSeq->inputs_count.emplace_back(1);
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
      taskDataSeq->inputs_count.emplace_back(1);
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
      taskDataSeq->inputs_count.emplace_back(1);
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
      taskDataSeq->outputs_count.emplace_back(expected_result.size());

      golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
      sequential_task.set_function([](double x) { return 5.0; });
      ASSERT_EQ(sequential_task.validation(), false);  // Валидация должна также провалиться
    }
  }
}
/* TEST(golovkin_integration_rectangular_method, test_linear_function) {
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 5.0;
  int partition_count = 1000000;

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return 2.0 * x + 3.0; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

 if (comm_world.rank() == 0) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return 2.0 * x + 3.0; });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_quadratic_function) {
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = -3.0;
  double upper_limit = 3.0;
  int partition_count = 1000000;

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return x * x; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();
  if (comm_world.rank() == 0) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return x * x; });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_sine_function) {
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = M_PI;
  int partition_count = 1000000;

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::sin(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (comm_world.rank() == 0) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::sin(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_cosine_function) {
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = M_PI / 2;
  int partition_count = 1000000;

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::cos(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (comm_world.rank() == 0) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::cos(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_exponential_function) {
  boost::mpi::communicator comm_world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 1.0;
  int partition_count = 1000000;

  if (comm_world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::exp(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (comm_world.rank() == 0) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::exp(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}*/