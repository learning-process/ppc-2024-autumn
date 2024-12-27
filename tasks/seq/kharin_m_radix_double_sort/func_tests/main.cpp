#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/kharin_m_radix_double_sort/include/ops_seq.hpp"

using namespace kharin_m_radix_double_sort;

// Тест на корректность параллельной и последовательной поразрядной сортировки на простом наборе данных
TEST(kharin_m_radix_double_sort_seq, SimpleData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 8;
  std::vector<double> inputData = {3.5, -2.1, 0.0, 1.1, -3.3, 2.2, -1.4, 5.6};
  std::vector<double> xSeq(N, 0.0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Последовательная версия
  RadixSortSequential radixSortSeq(taskDataSeq);
  ASSERT_TRUE(radixSortSeq.validation());
  radixSortSeq.pre_processing();
  radixSortSeq.run();
  radixSortSeq.post_processing();

  auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);
  // Сравнение со стандартной сортировкой
  std::sort(inputData.begin(), inputData.end());

  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(inputData[i], resultSeq[i], 1e-12);
  }
}

// Тест на некорректные входные данные (меньше данных, чем заявлено)
TEST(kharin_m_radix_double_sort_seq, ValidationFailureTestSize) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 5;
  std::vector<double> inputData = {3.5, -2.1, 0.0};  // всего 3 элемента, заявлено 5
  std::vector<double> xSeq(N, 0.0);

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

// Дополнительный тест: сортировка случайных double (малый размер)
TEST(kharin_m_radix_double_sort_seq, RandomDataSmall) {
  // Генерируем небольшой случайный массив double
  int N = 20;
  std::vector<double> inputData(N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  for (int i = 0; i < N; ++i) {
    inputData[i] = dist(gen);
  }

  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  RadixSortSequential radixSortSeq(taskDataSeq);
  ASSERT_TRUE(radixSortSeq.validation());
  radixSortSeq.pre_processing();
  radixSortSeq.run();
  radixSortSeq.post_processing();

  auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

  std::sort(inputData.begin(), inputData.end());

  // Проверяем, что результаты совпадают
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(inputData[i], resultSeq[i], 1e-12);
  }
}

// Дополнительный тест: сортировка большого массива случайных double
TEST(kharin_m_radix_double_sort_seq, RandomDataLarge) {
  // Генерируем небольшой случайный массив double
  int N = 10000;
  std::vector<double> inputData(N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  for (int i = 0; i < N; ++i) {
    inputData[i] = dist(gen);
  }

  std::vector<double> xSeq(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  RadixSortSequential radixSortSeq(taskDataSeq);
  ASSERT_TRUE(radixSortSeq.validation());
  radixSortSeq.pre_processing();
  radixSortSeq.run();
  radixSortSeq.post_processing();

  auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);

  std::sort(inputData.begin(), inputData.end());

  // Проверяем, что результаты совпадают
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(inputData[i], resultSeq[i], 1e-12);
  }
}

// Дополнительный тест: уже отсортированный массив
TEST(kharin_m_radix_double_sort_seq, AlreadySortedData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 10;
  std::vector<double> inputData = {-5.4, -3.3, -1.0, 0.0, 0.1, 1.2, 2.3, 2.4, 3.5, 10.0};
  std::vector<double> xSeq(N, 0.0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Последовательная версия
  RadixSortSequential radixSortSeq(taskDataSeq);
  ASSERT_TRUE(radixSortSeq.validation());
  radixSortSeq.pre_processing();
  radixSortSeq.run();
  radixSortSeq.post_processing();

  auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);
  // Сравнение со стандартной сортировкой
  std::sort(inputData.begin(), inputData.end());

  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(inputData[i], resultSeq[i], 1e-12);
  }
}

// Дополнительный тест: массив в обратном порядке
TEST(kharin_m_radix_double_sort_seq, ReverseSortedData) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  int N = 10;
  std::vector<double> inputData = {10.0, 3.5, 2.4, 2.3, 1.2, 0.1, 0.0, -1.0, -3.3, -5.4};
  std::vector<double> xSeq(N, 0.0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Последовательная версия
  RadixSortSequential radixSortSeq(taskDataSeq);
  ASSERT_TRUE(radixSortSeq.validation());
  radixSortSeq.pre_processing();
  radixSortSeq.run();
  radixSortSeq.post_processing();

  auto* resultSeq = reinterpret_cast<double*>(taskDataSeq->outputs[0]);
  // Сравнение со стандартной сортировкой
  std::sort(inputData.begin(), inputData.end());

  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(inputData[i], resultSeq[i], 1e-12);
  }
}