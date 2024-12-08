#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/kharin_m_radix_double_sort/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kharin_m_radix_double_sort;

// Тест на корректность параллельной и последовательной поразрядной сортировки на простом наборе данных
TEST(RadixSort_MPI, SimpleData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 8;
  std::vector<double> inputData = {3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6};
  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  // Параллельная версия
  RadixSortParallel radixSortPar(taskDataPar);
  ASSERT_TRUE(radixSortPar.validation());
  radixSortPar.pre_processing();
  radixSortPar.run();
  radixSortPar.post_processing();

  if (world.rank() == 0) {
    // Последовательная версия
    RadixSortSequential radixSortSeq(taskDataSeq);
    ASSERT_TRUE(radixSortSeq.validation());
    radixSortSeq.pre_processing();
    radixSortSeq.run();
    radixSortSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

// Тест на некорректные входные данные (меньше данных, чем заявлено)
TEST(RadixSort_MPI, ValidationFailureTestSize) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 5;
  std::vector<double> inputData = {3.5, -2.1, 0.0};  // всего 3 элемента, заявлено 5
  std::vector<double> xSeq(N, 0.0);

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataSeq->inputs_count.emplace_back(3);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);

    RadixSortSequential radixSortSeq(taskDataSeq);
    // Ожидаем провал валидации
    ASSERT_FALSE(radixSortSeq.validation());
  }
}

// Дополнительный тест: сортировка случайных double (малый размер)
TEST(RadixSort_MPI, RandomDataSmall) {
  mpi::environment env;
  mpi::communicator world;

  // Генерируем небольшой случайный массив double
  int N = 20;
  std::vector<double> inputData(N);
  if (world.rank() == 0) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  RadixSortParallel radixSortPar(taskDataPar);
  ASSERT_TRUE(radixSortPar.validation());
  radixSortPar.pre_processing();
  radixSortPar.run();
  radixSortPar.post_processing();

  if (world.rank() == 0) {
    RadixSortSequential radixSortSeq(taskDataSeq);
    ASSERT_TRUE(radixSortSeq.validation());
    radixSortSeq.pre_processing();
    radixSortSeq.run();
    radixSortSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    // Проверяем, что результаты совпадают
    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

// Дополнительный тест: сортировка большого массива случайных double
TEST(RadixSort_MPI, RandomDataLarge) {
  mpi::environment env;
  mpi::communicator world;

  // Генерируем большой случайный массив double
  int N = 10000;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData.resize(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  RadixSortParallel radixSortPar(taskDataPar);
  ASSERT_TRUE(radixSortPar.validation());
  radixSortPar.pre_processing();
  radixSortPar.run();
  radixSortPar.post_processing();

  if (world.rank() == 0) {
    RadixSortSequential radixSortSeq(taskDataSeq);
    ASSERT_TRUE(radixSortSeq.validation());
    radixSortSeq.pre_processing();
    radixSortSeq.run();
    radixSortSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    // Проверяем корректность результатов
    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

// Дополнительный тест: уже отсортированный массив
TEST(RadixSort_MPI, AlreadySortedData) {
  mpi::environment env;
  mpi::communicator world;

  int N = 10;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData = {-5.4, -3.3, -1.0, 0.0, 0.1, 1.2, 2.3, 2.4, 3.5, 10.0};
  }

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  RadixSortParallel radixSortPar(taskDataPar);
  ASSERT_TRUE(radixSortPar.validation());
  radixSortPar.pre_processing();
  radixSortPar.run();
  radixSortPar.post_processing();

  if (world.rank() == 0) {
    RadixSortSequential radixSortSeq(taskDataSeq);
    ASSERT_TRUE(radixSortSeq.validation());
    radixSortSeq.pre_processing();
    radixSortSeq.run();
    radixSortSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}

// Дополнительный тест: массив в обратном порядке
TEST(RadixSort_MPI, ReverseSortedData) {
  mpi::environment env;
  mpi::communicator world;

  int N = 10;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData = {10.0, 3.5, 2.4, 2.3, 1.2, 0.1, 0.0, -1.0, -3.3, -5.4};
  }

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  RadixSortParallel radixSortPar(taskDataPar);
  ASSERT_TRUE(radixSortPar.validation());
  radixSortPar.pre_processing();
  radixSortPar.run();
  radixSortPar.post_processing();

  if (world.rank() == 0) {
    RadixSortSequential radixSortSeq(taskDataSeq);
    ASSERT_TRUE(radixSortSeq.validation());
    radixSortSeq.pre_processing();
    radixSortSeq.run();
    radixSortSeq.post_processing();

    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-12);
    }
  }
}