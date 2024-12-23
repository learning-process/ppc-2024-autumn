#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <random>

#include "mpi/kharin_m_seidel_method/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kharin_m_seidel_method;

// Тест 1: Простые данные
TEST(kharin_m_seidel_method_tests_mpi, SimpleData) {
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
TEST(kharin_m_seidel_method_tests_mpi, ValidationFailureTestMatrixSize) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  std::vector<double> A = {4, 1, 2, 3, 5, 1, 1, 1, 3};
  std::vector<double> b = {15, 15, 10};

  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_FALSE(gaussSeidelPar.validation());
}

// Тест 3: Матрица не диагонально доминантная
TEST(kharin_m_seidel_method_tests_mpi, ValidationFailureTestNonDiagonallyDominant) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_FALSE(gaussSeidelPar.validation());
}

// Тест 4: Неправильное количество выходных данных
TEST(kharin_m_seidel_method_tests_mpi, ValidationFailureTestOutputCount) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  std::vector<double> A = {4, 1, 2, 0, 3, 5, 1, 1, 1, 1, 3, 2, 2, 0, 1, 4};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_FALSE(gaussSeidelPar.validation());
}

// Тест 5: Случайная диагонально доминантная матрица
TEST(kharin_m_seidel_method_tests_mpi, RandomDiagonallyDominantMatrixS) {
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
TEST(kharin_m_seidel_method_tests_mpi, ValidationFailureTestZerosDiagonally) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0};
  std::vector<double> b = {15, 15, 10, 10};

  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_FALSE(gaussSeidelPar.validation());
}

// Тест 7: Валидация параллельной версии с корректными данными
TEST(kharin_m_seidel_method_tests_mpi, ParallelValidationWithCorrectData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 3;
  double eps = 1e-6;

  // Диагонально доминантная матрица
  std::vector<double> A = {4, 1, 1, 2, 5, 1, 1, 1, 3};
  std::vector<double> b = {6, 14, 8};
  std::vector<double> xPar(N, 0.0);

  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_TRUE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_TRUE(gaussSeidelPar.validation());
}

// Тест 8: Валидация параллельной версии с некорректными данными (недостаточный ранг)
TEST(kharin_m_seidel_method_tests_mpi, ParallelValidationFailureInsufficientRank) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 3;
  double eps = 1e-6;

  // Матрица с недостаточным рангом
  std::vector<double> A = {2, 4, 2, 4, 8, 4, 1, 2, 1};
  std::vector<double> b = {4, 8, 2};

  std::vector<double> xPar(N, 0.0);
  if (world.rank() == 0) {
    // Настройка данных для последовательной версии
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

    // Копируем данные для параллельной версии
    taskDataPar->inputs = taskDataSeq->inputs;
    taskDataPar->inputs_count = taskDataSeq->inputs_count;

    // Настройка выходных данных для параллельной версии
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Проверка последовательной версии
  if (world.rank() == 0) {
    GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
    ASSERT_FALSE(gaussSeidelSeq.validation());
  }

  // Проверка параллельной версии
  GaussSeidelParallel gaussSeidelPar(taskDataPar);
  ASSERT_FALSE(gaussSeidelPar.validation());
}

// Тест 9: Случайная диагонально доминантная матрица M
TEST(kharin_m_seidel_method_tests_mpi, RandomDiagonallyDominantMatrixM) {
  // Параметры теста
  mpi::environment env;
  mpi::communicator world;
  int N = 20;         // Размер матрицы
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

// Тест 10: Случайная диагонально доминантная матрица L
TEST(kharin_m_seidel_method_tests_mpi, RandomDiagonallyDominantMatrixL) {
  // Параметры теста
  mpi::environment env;
  mpi::communicator world;
  int N = 50;         // Размер матрицы
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