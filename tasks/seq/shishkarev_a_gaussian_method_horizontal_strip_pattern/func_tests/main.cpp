#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

std::vector<std::vector<double>> generate_matrix(int size) {
  std::vector<std::vector<double>> matrix(size, std::vector<double>(size, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i][j] = dist(gen);
    }
  }

  return matrix;
}

std::vector<double> generate_vector_b(int size) {
  std::vector<double> vector_b(size, 0.0);
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    vector_b[i] = dist(gen);
  }

  return vector_b;
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq

TEST(Sequential_Operations, Test_2x2) {
  int size = 2;

  auto matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::generate_matrix(size);
  auto vector_b = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::generate_vector_b(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential testTaskSequential(
      taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed for 2x2 test.";
  testTaskSequential.pre_processing();

  auto start_time = std::chrono::high_resolution_clock::now();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Elapsed time for Test_2x2: " << elapsed_time.count() << " seconds." << std::endl;

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed after running for 2x2 test.";
}

TEST(Sequential_Operations, Test_5x5) {
  int size = 5;

  // Генерация матрицы 5x5 и вектора b
  auto matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::generate_matrix(size);
  auto vector_b = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::generate_vector_b(size);

  // Инициализация выходных данных
  std::vector<double> output_data(size, 0.0);

  // Создание объекта taskDataSeq и добавление данных
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  // Создание объекта тестируемой задачи
  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential testTaskSequential(
      taskDataSeq);

  // Проверка валидации
  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed for 5x5 test.";

  // Выполнение предварительной обработки
  testTaskSequential.pre_processing();

  // Замер времени выполнения
  auto start_time = std::chrono::high_resolution_clock::now();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  // Вывод времени выполнения
  std::cout << "Elapsed time for Test_5x5: " << elapsed_time.count() << " seconds." << std::endl;

  // Проверка валидации после выполнения
  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed after running for 5x5 test.";
}

TEST(Sequential_Operations, Test_invalid_data) {
  int size = 2;
  std::vector<std::vector<double>> matrix = {{2, 3}, {5, 4}, {1, 6}, {8, 9}};
  std::vector<double> vector_b = {1, 2, 3};

  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential testTaskSequential(
      taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(Sequential_Operations, Test_not_enough_data) {
  int size = 2;
  std::vector<std::vector<double>> matrix = {{2, 3}, {5, 4}};
  std::vector<double> vector_b = {1};

  std::vector<double> output_data(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussianHorizontalSequential testTaskSequential(
      taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}
