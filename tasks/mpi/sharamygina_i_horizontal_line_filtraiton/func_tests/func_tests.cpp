#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi\sharamygina_i_horizontal_line_filtraiton\include\ops_mpi.h"

namespace sharamygina_i_horizontal_line_filtration_mpi {
std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporaryIm(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) temporaryIm[i * cols + j] = dist(gen);
  return temporaryIm;
}

std::vector<unsigned int> ToFiltSeq(const std::vector<unsigned int>& image, int rows, int cols) {  // seq
  std::vector<unsigned int> final_image(rows * cols);
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  for (int x = 0; x < rows; x++)
    for (int y = 0; y < cols; y++) {
      unsigned int sum = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          int tX = x + i - 1, tY = y + j - 1;
          if (tX < 0 || tX > rows - 1) tX = x;
          if (tY < 0 || tY > cols - 1) tY = y;
          if (tX * cols + tY >= cols * rows) {
            tX = x;
            tY = y;
          }
          sum += static_cast<unsigned int>(image[tX * cols + tY] * (gauss[i][j]));
        }
      final_image[x * cols + y] = sum / 16;
    }
  return final_image;
}

}  // namespace sharamygina_i_horizontal_line_filtration_mpi

TEST(sharamygina_i_horizontal_line_filtration, SampleImageTest) {
  int rows = 4;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<unsigned int> expected_image = {0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0};

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(sharamygina_i_horizontal_line_filtration, BigImageTest) {
  boost::mpi::communicator world;

  int rows = 200;
  int cols = 160;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols, expected_image);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(sharamygina_i_horizontal_line_filtration, SmallImageTest) {
  boost::mpi::communicator world;

  int rows = 5;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols, expected_image);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(sharamygina_i_horizontal_line_filtration, SquareImageTest) {
  boost::mpi::communicator world;

  int rows = 5;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols, expected_image);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(sharamygina_i_horizontal_line_filtration, HorizontalImageTest) {
  boost::mpi::communicator world;

  int rows = 5;
  int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols, expected_image);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(sharamygina_i_horizontal_line_filtration, VerticalImageTest) {
  boost::mpi::communicator world;

  int rows = 10;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(rows);
  taskData->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  if (world.rank() == 0) {
    image = sharamygina_i_horizontal_line_filtration_mpi::GetImage(rows, cols);
    sharamygina_i_horizontal_line_filtration_mpi::ToFiltSeq(image, rows, cols, expected_image);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

    received_image.resize(rows * cols);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
    taskData->outputs_count.emplace_back(received_image.size());
  }

  sharamygina_i_horizontal_line_filtration_mpi::horizontal_line_filtration_mpi testTask(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (size_t i = 0; i < received_image.size(); i++) {
      ASSERT_EQ(received_image[i], expected[i]) << "Difference at i=" << i;
    }
  }
}