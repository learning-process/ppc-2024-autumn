#include <gtest/gtest.h>

#include <algorithm>  // для std::sort
#include <boost/mpi.hpp>
#include <memory>
#include <random>
#include <vector>

#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

// Используем namespace для MPI варианта
using namespace petrov_o_radix_sort_with_simple_merge_mpi;

// Базовый тест на сортировку небольшого массива
TEST(petrov_o_radix_sort_with_simple_merge_mpi, BasicSortTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Исходные данные на rank=0
  std::vector<int> in, out;
  if (world.rank() == 0) {
    in = {8, 3};
    out.resize(in.size(), 0);
  }

  // Создание TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  } else {
    // На других рангах данные не инициализируем (пусть будут пустые)
    taskData->inputs_count.emplace_back(0);
    taskData->outputs_count.emplace_back(0);
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  // Проверка результата только на rank=0
  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

// Тестирование сортировки массива с отрицательными числами
TEST(petrov_o_radix_sort_with_simple_merge_mpi, NegativeNumbersTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in, out;
  if (world.rank() == 0) {
    in = {-100, -5, -3, 2, 7, 12};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  } else {
    taskData->inputs_count.emplace_back(0);
    taskData->outputs_count.emplace_back(0);
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

// Тестирование сортировки массива, отсортированного в обратном порядке
TEST(petrov_o_radix_sort_with_simple_merge_mpi, ReverseSortedTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in, out;
  if (world.rank() == 0) {
    in = {10, 8, 6, 4, 2, 0};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  } else {
    taskData->inputs_count.emplace_back(0);
    taskData->outputs_count.emplace_back(0);
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

// Тестирование сортировки массива с одинаковыми элементами
TEST(petrov_o_radix_sort_with_simple_merge_mpi, DuplicateElementsTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in, out;
  if (world.rank() == 0) {
    in = {5, 5, 5, 5, 5, 5};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  } else {
    taskData->inputs_count.emplace_back(0);
    taskData->outputs_count.emplace_back(0);
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

// Тестирование сортировки массива со случайными значениями
TEST(petrov_o_radix_sort_with_simple_merge_mpi, RandomValuesTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in, out;
  if (world.rank() == 0) {
    // Задаём размер массива
    const size_t array_size = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    // Генерируем случайные числа
    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(gen);
    }

    // Выделяем место для отсортированного массива
    out.resize(in.size(), 0);
  }

  // Создание TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  } else {
    // На других рангах данные не инициализируем (пусть будут пустые)
    taskData->inputs_count.emplace_back(0);
    taskData->outputs_count.emplace_back(0);
  }

  // Создание и запуск задачи
  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation()) << "Validation failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.pre_processing()) << "Pre-processing failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.run()) << "Run failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.post_processing()) << "Post-processing failed on rank " << world.rank();

  // Проверка результата только на rank=0
  if (world.rank() == 0) {
    // Создаём копию исходного массива для сравнения
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    // Проверяем, что отсортированный результат совпадает с ожидаемым
    ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
  }

  // Синхронизация всех процессов
  world.barrier();
}

// Базовый тест на сортировку небольшого массива
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, BasicSortTest) {
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
  TaskSequential testTaskSequential(taskDataSeq);

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
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, NegativeNumbersTest) {
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
  TaskSequential testTaskSequential(taskDataSeq);

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
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, ReverseSortedTest) {
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
  TaskSequential testTaskSequential(taskDataSeq);

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
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, DuplicateElementsTest) {
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
  TaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  // Проверка результата
  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}