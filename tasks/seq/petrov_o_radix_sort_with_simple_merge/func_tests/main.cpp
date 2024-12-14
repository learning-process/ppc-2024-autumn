#include <gtest/gtest.h>

#include <vector>
#include <algorithm>  // для std::sort
#include <memory>

#include "seq/petrov_o_radix_sort_with_simple_merge/include/ops_seq.hpp"

// Базовый тест на сортировку небольшого массива
TEST(RadixSortTest, BasicSortTest) {
  // Создание данных
  std::vector<int> in{8, 3};
  std::vector<int> out(in.size(), 0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

// Тестирование сортировки массива с отрицательными числами
TEST(RadixSortTest, NegativeNumbersTest) {
  // Создание данных
  std::vector<int> in{-100, -5, -3, 2, 7, 12};
  std::vector<int> out(in.size(), 0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

// Тестирование сортировки массива, отсортированного в обратном порядке
TEST(RadixSortTest, ReverseSortedTest) {
  // Создание данных
  std::vector<int> in{10, 8, 6, 4, 2, 0};
  std::vector<int> out(in.size(), 0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

// Тестирование сортировки массива с одинаковыми элементами
TEST(RadixSortTest, DuplicateElementsTest) {
  // Создание данных
  std::vector<int> in{5, 5, 5, 5, 5, 5};
  std::vector<int> out(in.size(), 0);

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Создание задачи
  petrov_o_radix_sort_with_simple_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}