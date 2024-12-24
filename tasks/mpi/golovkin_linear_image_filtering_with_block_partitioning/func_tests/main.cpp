// Golovkin Maksim Task#3
#include <gtest/gtest.h>

#include "mpi/golovkin_linear_image_filtering_with_block_partitioning/include/ops_mpi.hpp"

using namespace std;

static void gauss_3x3(const std::vector<int>& input, int width, int height, std::vector<int>* output) {
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  int kernel_sum = 16;

  output->resize(width * height);

  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int sum = 0;

      for (int kr = -1; kr <= 1; kr++) {
        for (int kc = -1; kc <= 1; kc++) {
          int rr = r + kr;
          int cc = c + kc;

          if (rr >= 0 && rr < height && cc >= 0 && cc < width) {
            int kernel_value = kernel[kr + 1][kc + 1];
            int input_value = input[rr * width + cc];
            sum += input_value * kernel_value;
          }
        }
      }

      (*output)[r * width + c] = sum / kernel_sum;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, SmallImageBlockTest) {
  boost::mpi::communicator world;

  int width = 5;
  int height = 4;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterBlockOddDimensions) {
  boost::mpi::communicator world;

  int width = 7;
  int height = 5;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 1);
  vector<int> output(width * height, 0);

  shared_ptr<ppc::core::TaskData> taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TestGaussianFilterBlockUnevenDistribution) {
  boost::mpi::communicator world;

  int width = 4;
  int height = 7;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  shared_ptr<ppc::core::TaskData> taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}
TEST(golovkin_linear_image_filtering_with_block_partitioning, MinimalImageBlockTest) {
  boost::mpi::communicator world;
  int width = 3;
  int height = 3;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, NarrowImageBlockTest) {
  boost::mpi::communicator world;
  int width = 8;
  int height = 3;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, TallImageBlockTest) {
  boost::mpi::communicator world;
  int width = 3;
  int height = 8;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, SquareImageBlockTest) {
  boost::mpi::communicator world;
  int width = 6;
  int height = 6;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, LargerImageBlockTest) {
  boost::mpi::communicator world;
  int width = 10;
  int height = 10;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, SingleProcessTest) {
  boost::mpi::communicator world;
  int width = 4;
  int height = 5;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, VeryLargeImageTest) {
  boost::mpi::communicator world;

  int width = 500;
  int height = 500;

  vector<int> input(width * height);
  iota(input.begin(), input.end(), 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, ZeroImageTest) {
  boost::mpi::communicator world;

  int width = 8;
  int height = 8;

  vector<int> input(width * height, 0);
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    for (int val : output) {
      ASSERT_EQ(val, 0);
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, NegativePixelsTest) {
  boost::mpi::communicator world;

  int width = 5;
  int height = 5;

  vector<int> input(width * height);
  for (int i = 0; i < width * height; i++) {
    input[i] = -i;
  }
  vector<int> output(width * height, 0);

  auto taskData = make_shared<ppc::core::TaskData>();
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

  auto task = make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}

TEST(golovkin_linear_image_filtering_with_block_partitioning, VeryLargeImageTest_499x501) {
  boost::mpi::communicator world;

  int width = 499;
  int height = 501;

  std::vector<int> input(width * height);
  std::iota(input.begin(), input.end(), 0);
  std::vector<int> output(width * height, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
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

  auto task = std::make_shared<golovkin_linear_image_filtering_with_block_partitioning::SimpleBlockMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    // Проверяем корректность результата
    std::vector<int> expected;
    gauss_3x3(input, width, height, &expected);
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); i++) {
      ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
    }
  }
}