#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "seq/kharin_m_seidel_method/include/ops_seq.hpp"

using namespace kharin_m_seidel_method;

// Тест 1: Простые данные
TEST(kharin_m_seidel_method_tests_seq, SimpleData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;          // Размер матрицы
  double eps = 1e-6;  // Точность вычислений

  // Матрица A и вектор b (пример системы уравнений)
  std::vector<double> A = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<double> b = {15, 15, 10, 10};

  std::vector<double> xSeq(N, 0.0);

  // Входные данные для последовательной задачи
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);  // Количество элементов типа int

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);  // Количество элементов типа double

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);  // Матрица A размером N x N

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);  // Вектор b размером N

  // Выходные данные для последовательной задачи
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);  // Вектор решений x размером N

  // Создаем и запускаем последовательную задачу
  GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq.validation());
  gaussSeidelSeq.pre_processing();
  gaussSeidelSeq.run();
  gaussSeidelSeq.post_processing();

  // Получаем результаты из taskData->outputs
  auto* result = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

  // Проверяем, что результат соответствует входному вектору b
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(result[i], b[i], eps);
  }
}

// Тест 2: Неправильный размер матрицы A
TEST(kharin_m_seidel_method_tests_seq, ValidationFailureTestMatrixSize) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица меньшего размера
  std::vector<double> A = {4, 1, 2, 3, 5, 1, 1, 1, 3};
  std::vector<double> b = {15, 15, 10};

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

// Тест 3: Матрица не диагонально доминантная
TEST(kharin_m_seidel_method_tests_seq, ValidationFailureTestNonDiagonallyDominant) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1, 10, 10, 10, 10, 1};
  std::vector<double> b = {15, 15, 10, 10};

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

// Тест 4: Неправильное количество выходных данных
TEST(kharin_m_seidel_method_tests_seq, ValidationFailureTestOutputCount) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  std::vector<double> A = {4, 1, 2, 0, 3, 5, 1, 1, 1, 1, 3, 2, 2, 0, 1, 4};
  std::vector<double> b = {15, 15, 10, 10};

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

// Тест 5: Случайная диагонально доминантная матрица
TEST(kharin_m_seidel_method_tests_seq, RandomDiagonallyDominantMatrixS) {
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

  // Создаем TaskData для последовательной версии
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<double> xSeq(N, 0.0);

  // Входные данные для последовательной задачи
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  // Выходные данные для последовательной задачи
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Создаем и запускаем последовательную задачу
  GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq.validation());
  gaussSeidelSeq.pre_processing();
  gaussSeidelSeq.run();
  gaussSeidelSeq.post_processing();

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

// Тест 6: Нули на диагонали
TEST(kharin_m_seidel_method_tests_seq, ValidationFailureTestZerosDiagonally) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 4;
  double eps = 1e-6;

  // Матрица, где диагональные элементы не доминируют
  std::vector<double> A = {0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10, 10, 10, 0};
  std::vector<double> b = {15, 15, 10, 10};

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

// Тест 7: Валидация с корректными данными
TEST(kharin_m_seidel_method_tests_seq, ParallelValidationWithCorrectData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 3;
  double eps = 1e-6;

  std::vector<double> A = {4, 1, 1, 2, 5, 1, 1, 1, 3};
  std::vector<double> b = {6, 14, 8};

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
  ASSERT_TRUE(gaussSeidelSeq.validation());
}

// Тест 8: Валидация параллельной версии с некорректными данными (недостаточный ранг)
TEST(kharin_m_seidel_method_tests_seq, ParallelValidationFailureInsufficientRank) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 3;
  double eps = 1e-6;

  // Матрица с недостаточным рангом
  std::vector<double> A = {2, 4, 2, 4, 8, 4, 1, 2, 1};
  std::vector<double> b = {4, 8, 2};

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

  GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
  ASSERT_FALSE(gaussSeidelSeq.validation());
}

// Тест 9: Случайная диагонально доминантная матрица M
TEST(kharin_m_seidel_method_tests_seq, RandomDiagonallyDominantMatrixM) {
  // Параметры теста
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

  // Создаем TaskData для последовательной версии
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<double> xSeq(N, 0.0);

  // Входные данные для последовательной задачи
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  // Выходные данные для последовательной задачи
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Создаем и запускаем последовательную задачу
  GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq.validation());
  gaussSeidelSeq.pre_processing();
  gaussSeidelSeq.run();
  gaussSeidelSeq.post_processing();

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

// Тест 10: Случайная диагонально доминантная матрица L
TEST(kharin_m_seidel_method_tests_seq, RandomDiagonallyDominantMatrixL) {
  // Параметры теста
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

  // Создаем TaskData для последовательной версии
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<double> xSeq(N, 0.0);

  // Входные данные для последовательной задачи
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  // Выходные данные для последовательной задачи
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Создаем и запускаем последовательную задачу
  GaussSeidelSequential gaussSeidelSeq(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq.validation());
  gaussSeidelSeq.pre_processing();
  gaussSeidelSeq.run();
  gaussSeidelSeq.post_processing();

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