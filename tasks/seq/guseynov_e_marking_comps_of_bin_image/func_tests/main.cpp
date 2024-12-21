#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/guseynov_e_marking_comps_of_bin_image/include/ops_seq.hpp"

std::vector<int> getRandomBinImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = gen() % 2;
  }
  return vec;
}

void checkNeighbors(const std::vector<int> &matrix, int rows, int cols) {
  // Направления: (dx, dy)
  std::vector<std::pair<int, int>> directions = {
      {0, 1},   // вправо
      {0, -1},  // влево
      {1, 0},   // вниз
      {-1, 0},  // вверх
      {-1, 1},  // по диагонали вниз вправо
      {1, -1}   // по диагонали вверх влево
  };

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int currentIndex = i * cols + j;
      int currentValue = matrix[currentIndex];

      // Пропускаем элемент, если он равен 1
      if (currentValue == 1) {
        continue;
      }

      bool shouldPrint = false;

      // Проверяем соседей
      for (const auto &dir : directions) {
        int newRow = i + dir.first;
        int newCol = j + dir.second;

        // Проверяем границы матрицы
        if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
          int neighborIndex = newRow * cols + newCol;
          int neighborValue = matrix[neighborIndex];

          // Если сосед не равен текущему элементу и не равен 1
          if (neighborValue != currentValue && neighborValue != 1) {
            shouldPrint = true;
            break;  // Достаточно одного такого соседа
          }
        }
      }

      // Если условие выполнено, выводим элемент с запятой
      if (shouldPrint) {
        std::cout << "[" << i << ", " << j << ", " << currentValue << "]";
      }
    }
  }
  std::cout << std::endl;  // Завершаем строку
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_image_is_object) {
  const int rows = 3;
  const int columns = 3;
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_image_is_background) {
  const int rows = 3;
  const int columns = 3;
  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_with_isolated_points) {
  const int rows = 3;
  const int columns = 3;
  std::vector<int> in = {0, 1, 1, 1, 1, 0, 0, 1, 1};
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out = {2, 1, 1, 1, 1, 3, 4, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_with_no_isolated_points) {
  const int rows = 3;
  const int columns = 3;
  std::vector<int> in = {0, 0, 0, 1, 1, 1, 0, 0, 1};
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out = {2, 2, 2, 1, 1, 1, 3, 3, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_one_row_with_isolated_points) {
  const int rows = 1;
  const int columns = 3;
  std::vector<int> in = {0, 1, 0};
  std::vector<int> out(rows * columns);
  std::vector<int> expected_out = {2, 1, 3};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_one_column_with_isolated_point) {
  const int rows = 3;
  const int columns = 1;
  std::vector<int> in = {0, 1, 0};
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out = {2, 1, 3};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_one_column_with_isolated_point2) {
  const int rows = 10;
  const int columns = 10;
  std::vector<int> in = getRandomBinImage(rows, columns);
  std::vector<int> out(rows * columns, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}
