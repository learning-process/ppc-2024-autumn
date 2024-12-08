// gauss_seidel_mpi_test.cpp
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <random>

#include "mpi/kharin_m_seidel_method/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kharin_m_seidel_method;

// Тест 1: Простые данные
TEST(GaussSeidel_MPI, SimpleData) {
  mpi::environment env;
  mpi::communicator world;
  // Создаем TaskData для параллельной и последовательной версий
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;          // Размер матрицы
  double eps = 1e-6;  // Точность вычислений

  // Матрица A и вектор b (пример системы уравнений)
  std::vector<double> A = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<double> b = {15, 15, 10, 10};

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  // Инициализируем входные данные и результаты на процессе 0
  if (world.rank() == 0) {
    // Входные данные для параллельной задачи
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);  // Количество элементов типа int

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);  // Количество элементов типа double

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N * N);  // Матрица A размером N x N

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(N);  // Вектор b размером N

    // Выходные данные для параллельной задачи
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);  // Вектор решений x размером N

    // Входные данные для последовательной задачи (идентичны параллельной)
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    // Выходные данные для последовательной задачи
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  // Создаем и запускаем параллельную задачу
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_TRUE(gaussSeidelPar.validation());
  gaussSeidelPar.pre_processing();
  gaussSeidelPar.run();
  gaussSeidelPar.post_processing();

  // Создаем и запускаем последовательную задачу на процессе 0
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_TRUE(gaussSeidelSeq.validation());
    gaussSeidelSeq.pre_processing();
    gaussSeidelSeq.run();
    gaussSeidelSeq.post_processing();

    // Получаем результаты из taskData->outputs
    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);
    size_t sizePar = taskDataPar->outputs_count[0];
    size_t sizeSeq = taskDataSeq->outputs_count[0];

    // Сравниваем размеры выходных данных
    ASSERT_EQ(sizePar, sizeSeq);

    // Сравниваем результаты
    for (size_t i = 0; i < sizePar; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-6);
    }
  }
}

// Тест 2: Неправильный размер матрицы A
TEST(GaussSeidel_MPI, ValidationFailureTestMatrixSize) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица меньшего размера
  std::vector<double> A = {4, 1, 2, 3, 5, 1, 1, 1, 3};
  std::vector<double> b = {15, 15, 10};

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    // Намеренно указываем неправильный размер
    taskDataSeq->inputs_count.emplace_back(3 * 3);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.emplace_back(3);

    std::vector<double> xSeq(N, 0.0);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);

    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }
}

// Тест 3: Матрица не диагонально доминантная
TEST(GaussSeidel_MPI, ValidationFailureTestNonDiagonallyDominant) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N * N);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.emplace_back(N);

    std::vector<double> xSeq(N, 0.0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);

    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }
}

// Тест 4: Неправильное количество выходных данных
TEST(GaussSeidel_MPI, ValidationFailureTestOutputCount) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  std::vector<double> A = {4, 1, 2, 0, 3, 5, 1, 1, 1, 1, 3, 2, 2, 0, 1, 4};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N * N);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.emplace_back(N);

    std::vector<double> xSeq1(N, 0.0);
    std::vector<double> xSeq2(N, 0.0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq1.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq2.data()));
    taskDataSeq->outputs_count.emplace_back(N);
    taskDataSeq->outputs_count.emplace_back(N);

    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }
}

// Тест 5: Случайная диагонально доминантная матрица
TEST(GaussSeidel_MPI, RandomDiagonallyDominantMatrix) {
  mpi::environment env;
  mpi::communicator world;

  // Параметры теста
  int N = 6;          // Размер матрицы
  double eps = 1e-6;  // Точность вычислений

  // Создаем генератор случайных чисел
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  // Создаем случайную диагонально доминантную матрицу
  std::vector<double> A(N * N);
  std::vector<double> b(N);

  for (int i = 0; i < N; ++i) {
    // Сумма абсолютных значений недиагональных элементов
    double offDiagonalSum = 0.0;

    for (int j = 0; j < N; ++j) {
      if (i == j) continue;
      A[i * N + j] = dis(gen);
      offDiagonalSum += std::abs(A[i * N + j]);
    }

    // Диагональный элемент должен быть больше суммы остальных
    A[i * N + i] = offDiagonalSum + std::abs(dis(gen));

    // Случайный вектор b
    b[i] = dis(gen);
  }

  // Создаем TaskData для параллельной и последовательной версий
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<double> xPar(N, 0.0);
  std::vector<double> xSeq(N, 0.0);

  // Инициализируем входные данные и результаты на процессе 0
  if (world.rank() == 0) {
    // Входные данные для параллельной задачи
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N * N);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(N);

    // Выходные данные для параллельной задачи
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);

    // Входные данные для последовательной задачи (идентичны параллельной)
    taskDataSeq->inputs = taskDataPar->inputs;
    taskDataSeq->inputs_count = taskDataPar->inputs_count;

    // Выходные данные для последовательной задачи
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);
  }

  // Создаем и запускаем параллельную задачу
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_TRUE(gaussSeidelPar.validation());
  gaussSeidelPar.pre_processing();
  gaussSeidelPar.run();
  gaussSeidelPar.post_processing();

  // Создаем и запускаем последовательную задачу на процессе 0
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_TRUE(gaussSeidelSeq.validation());
    gaussSeidelSeq.pre_processing();
    gaussSeidelSeq.run();
    gaussSeidelSeq.post_processing();

    // Получаем результаты из taskData->outputs
    auto* resultPar = reinterpret_cast<double*>(taskDataPar->outputs[0]);
    auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);
    size_t sizePar = taskDataPar->outputs_count[0];
    size_t sizeSeq = taskDataSeq->outputs_count[0];

    // Сравниваем размеры выходных данных
    ASSERT_EQ(sizePar, sizeSeq);

    // Сравниваем результаты
    for (size_t i = 0; i < sizePar; ++i) {
      ASSERT_NEAR(resultPar[i], resultSeq[i], 1e-6);
    }

    // Дополнительная проверка: генерация случайной матрицы прошла успешно
    ASSERT_NO_THROW({
      // Проверка диагональной доминантности
      for (int i = 0; i < N; ++i) {
        double diagonalElement = std::abs(A[i * N + i]);
        double offDiagonalSum = 0.0;
        for (int j = 0; j < N; ++j) {
          if (i != j) {
            offDiagonalSum += std::abs(A[i * N + j]);
          }
        }
        EXPECT_GT(diagonalElement, offDiagonalSum);
      }
    });
  }
}

// Тест 6: Нули на диагонали
TEST(GaussSeidel_MPI, ValidationFailureTestZerosDiagonally) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs_count.emplace_back(N * N);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs_count.emplace_back(N);

    std::vector<double> xSeq(N, 0.0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataSeq->outputs_count.emplace_back(N);

    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }
}