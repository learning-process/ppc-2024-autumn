#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"
#include "mpi/yasakova_t_quick_sort_with_simple_merge/include/ops_mpi.hpp"

namespace yasakova_t_quick_sort_with_simple_merge_mpi {

std::vector<int> create_random_integer_vector(int size, int minimum_value = -100, int maximum_value = 100,
                                              unsigned random_seed = std::random_device{}()) {
    static std::mt19937 generator(random_seed); // Генератор случайных чисел
    std::uniform_int_distribution<int> distribution(minimum_value, maximum_value); // Распределение

    std::vector<int> random_vector(size); // Вектор для хранения случайных чисел
    std::generate(random_vector.begin(), random_vector.end(), [&]() { return distribution(generator); }); // Заполнение вектора случайными числами
    return random_vector; // Возврат сгенерированного вектора
}


void execute_parallel_sort_test(int vector_length, bool ascending = true) {
  boost::mpi::communicator mpi_comm;
  std::vector<int> input_vector;
  std::vector<int> output_vector;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_length));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    input_vector = create_random_integer_vector(vector_length);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());

    output_vector.resize(vector_length);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  auto parallel_sort_task = std::make_shared<SimpleMergeQuicksort>(task_data);

  // Установите порядок сортировки
  parallel_sort_task->set_sort_order(ascending);

  bool is_valid = parallel_sort_task->validation();
  boost::mpi::broadcast(mpi_comm, is_valid, 0);
  if (is_valid) {
    parallel_sort_task->pre_processing();
    parallel_sort_task->run();
    parallel_sort_task->post_processing();

    if (mpi_comm.rank() == 0) {
      // Сортировка для проверки
      if (ascending) {
        std::sort(input_vector.begin(), input_vector.end());
      } else {
        std::sort(input_vector.begin(), input_vector.end(), std::greater<int>());
      }
      EXPECT_EQ(input_vector, output_vector);
    }
  }
}

void execute_parallel_sort_test(const std::vector<int>& input_vector, bool ascending = true) {
  boost::mpi::communicator mpi_comm;
  std::vector<int> local_data = input_vector;
  std::vector<int> sorted_data;

  int vector_size = static_cast<int>(local_data.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_data.data()));
    task_data->inputs_count.emplace_back(local_data.size());

    sorted_data.resize(vector_size);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_data.data()));
    task_data->outputs_count.emplace_back(sorted_data.size());
  }

  auto parallel_sort_task = std::make_shared<SimpleMergeQuicksort>(task_data);

  // Установите порядок сортировки
  parallel_sort_task->set_sort_order(ascending);

  bool is_valid = parallel_sort_task->validation();
  boost::mpi::broadcast(mpi_comm, is_valid, 0);
  if (is_valid) {
    parallel_sort_task->pre_processing();
    parallel_sort_task->run();
    parallel_sort_task->post_processing();

    if (mpi_comm.rank() == 0) {
      // Сортировка для проверки
      if (ascending) {
        std::sort(local_data.begin(), local_data.end());
      } else {
        std::sort(local_data.begin(), local_data.end(), std::greater<int>());
      }
      EXPECT_EQ(local_data, sorted_data);
    }
  }
}

}  // namespace yasakova_t_quick_sort_with_simple_merge_mpi

// Тест для сортировки по возрастанию
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_sorted_array_ascending) {
  std::vector<int> input_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
}

// Тест для сортировки по убыванию
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_sorted_array_descending) {
  std::vector<int> input_vector = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для случайного массива
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_random_array) {
  int vector_length = 100; // Длина случайного массива
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(vector_length, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(vector_length, false); // Сортировка по убыванию
}

// Тест для пустого массива
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_empty_array) {
  std::vector<int> input_vector = {};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для массива с одинаковыми элементами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_same_elements) {
  std::vector<int> input_vector = {5, 5, 5, 5, 5};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}
// Тест для сортировки массива с отрицательными числами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_negative_numbers) {
  std::vector<int> input_vector = {-5, -1, -3, -4, -2};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для сортировки массива с положительными числами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_positive_numbers) {
  std::vector<int> input_vector = {5, 1, 3, 4, 2};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для сортировки массива с смешанными числами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_mixed_numbers) {
  std::vector<int> input_vector = {3, -1, 4, 1, -5, 9, 2, -6};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для сортировки массива с большим количеством элементов
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_large_array) {
  int vector_length = 1000; // Длина случайного массива
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(vector_length, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(vector_length, false); // Сортировка по убыванию
}

// Тест для сортировки массива с повторяющимися элементами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_repeated_elements) {
  std::vector<int> input_vector = {1, 2, 2, 3, 1, 4, 4, 4, 5};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для сортировки массива с одним элементом
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_single_element) {
  std::vector<int> input_vector = {42};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, true); // Сортировка по возрастанию
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector, false); // Сортировка по убыванию
}

// Тест для сортировки массива с двумя элементами
TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_two_elements) {
  std::vector<int> input_vector_asc = {2, 1};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector_asc, true); // Сортировка по возрастанию

  std::vector<int> input_vector_desc = {1, 2};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(input_vector_desc, false); // Сортировка по убыванию
}