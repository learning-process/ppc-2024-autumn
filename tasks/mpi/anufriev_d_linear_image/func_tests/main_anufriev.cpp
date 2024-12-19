#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"

static void gaussian_3x3_seq(const std::vector<int>& input, int width, int height, std::vector<int>* output) {
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  output->resize(width * height);

  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int sum = 0;
      for (int kr = -1; kr <= 1; kr++) {
        for (int kc = -1; kc <= 1; kc++) {
          int rr = std::min(std::max(r + kr, 0), height - 1);
          int cc = std::min(std::max(c + kc, 0), width - 1);
          sum += input[rr * width + cc] * kernel[kr + 1][kc + 1];
        }
      }
      (*output)[r * width + c] = sum / 16;
    }
  }
}

TEST(anufriev_d_linear_image_func_mpi, SmallImageTest) {
  boost::mpi::communicator world;

  int width = 5;
  int height = 4;

  std::vector<int> input(width * height);
  std::iota(input.begin(), input.end(), 0);
  std::vector<int> output(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size() * sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&width));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&height));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size() * sizeof(int));
  }

  auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected;
    gaussian_3x3_seq(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(anufriev_d_linear_image_func_mpi, LargerImageRandomTest) {
  boost::mpi::communicator world;

  int width = 100;
  int height = 80;

  std::vector<int> input;
  std::vector<int> output;
  if (world.rank() == 0) {
    input.resize(width * height);
    srand(123);
    for (auto& val : input) val = rand() % 256;
    output.resize(width * height, 0);
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size() * sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&width));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&height));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size() * sizeof(int));
  }

  auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected;
    gaussian_3x3_seq(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(anufriev_d_linear_image_func_mpi, TestGaussianFilterOddDimensions) {
  boost::mpi::communicator world;

  int width = 7;
  int height = 5;

  std::vector<int> input(width * height);
  std::iota(input.begin(), input.end(), 1);
  std::vector<int> output(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size() * sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&width));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&height));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size() * sizeof(int));
  }

  auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected;
    gaussian_3x3_seq(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(anufriev_d_linear_image_func_mpi, TestGaussianFilterUnevenDistribution) {
  boost::mpi::communicator world;

  int width = 4;
  int height = 7;

  std::vector<int> input(width * height);
  std::iota(input.begin(), input.end(), 0);
  std::vector<int> output(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size() * sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&width));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&height));
    taskData->inputs_count.push_back(sizeof(int));

    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size() * sizeof(int));
  }

  auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected;
    gaussian_3x3_seq(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}