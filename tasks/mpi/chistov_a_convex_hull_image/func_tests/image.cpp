#include "mpi/chistov_a_convex_hull_image/include/image.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

namespace chistov_a_convex_hull_image_mpi_test {
std::vector<int> generateImage(int width, int height) {
  if (width <= 0 || height <= 0) {
    return {};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1);

  std::vector<int> image(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      image[y * width + x] = dist(gen);
    }
  }

  return image;
}
}  // namespace chistov_a_convex_hull_image_mpi_test

TEST(chistov_a_convex_hull_image_mpi, validation_test_zero_height_and_width) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int width = 0;
    const int height = 0;
    std::vector<int> image = chistov_a_convex_hull_image_mpi_test::generateImage(width, height);
    std::vector<int> hull(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(chistov_a_convex_hull_image_mpi, validation_test_empty_vector) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int width = 3;
    const int height = 4;
    std::vector<int> image;
    std::vector<int> hull;
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(chistov_a_convex_hull_image_mpi, validation_test_negative_size) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int width = -1;
    const int height = -1;
    std::vector<int> image = chistov_a_convex_hull_image_mpi_test::generateImage(width, height);
    std::vector<int> hull(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(chistov_a_convex_hull_image_mpi, validation_test_empty_output) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int width = 10;
    const int height = 10;
    std::vector<int> image = chistov_a_convex_hull_image_mpi_test::generateImage(width, height);
    std::vector<int> hull(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs_count.emplace_back(width * height);

    chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(chistov_a_convex_hull_image_mpi, validation_not_binary_image) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int width = 10;
    const int height = 10;
    std::vector<int> image = chistov_a_convex_hull_image_mpi_test::generateImage(width, height);
    image[0] = 5;
    std::vector<int> hull(width * height);
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs_count.emplace_back(width * height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);

    chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
    ASSERT_FALSE(TestTaskMPI.validation());
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_image_of_zeros) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }
  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(image, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_single_points_image) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;
  std::vector<int> image(width * height, 0);
  std::vector<int> excepted_hull(width * height, 0);
  image[5 * width + 5] = 1;

  std::vector<int> hull(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);

  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(excepted_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_non_adjacent_points) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  image[0] = 1;
  image[1 * width + 1] = 1;
  image[3 * width + 3] = 1;
  image[5 * width + 5] = 1;
  image[7 * width + 7] = 1;

  std::vector<int> hull(width * height, 0);

  std::vector<int> excepted_hull = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
                                    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);

  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(excepted_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_one_component_image) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  image[2 * width + 1] = 1;
  image[2 * width + 2] = 1;
  image[2 * width + 3] = 1;
  image[3 * width + 1] = 1;
  image[3 * width + 2] = 1;
  image[3 * width + 3] = 1;
  image[4 * width + 1] = 1;
  image[4 * width + 2] = 1;
  image[4 * width + 3] = 1;

  std::vector<int> hull(width * height, 0);
  std::vector<int> excepted_hull = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(excepted_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_two_components_image) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  image[2 * width + 1] = 1;
  image[2 * width + 2] = 1;
  image[2 * width + 3] = 1;
  image[3 * width + 1] = 1;
  image[3 * width + 2] = 1;
  image[3 * width + 3] = 1;
  image[4 * width + 1] = 1;
  image[4 * width + 2] = 1;
  image[4 * width + 3] = 1;

  image[7 * width + 1] = 1;
  image[7 * width + 2] = 1;
  image[7 * width + 3] = 1;
  image[8 * width + 1] = 1;
  image[8 * width + 2] = 1;
  image[8 * width + 3] = 1;
  image[9 * width + 1] = 1;
  image[9 * width + 2] = 1;
  image[9 * width + 3] = 1;

  std::vector<int> expected_hull = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                                    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                                    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> hull(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_three_components_image) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  image[2 * width + 1] = 1;
  image[2 * width + 2] = 1;
  image[2 * width + 3] = 1;
  image[3 * width + 1] = 1;
  image[3 * width + 2] = 1;
  image[3 * width + 3] = 1;
  image[4 * width + 1] = 1;
  image[4 * width + 2] = 1;
  image[4 * width + 3] = 1;

  image[7 * width + 1] = 1;
  image[7 * width + 2] = 1;
  image[7 * width + 3] = 1;
  image[8 * width + 1] = 1;
  image[8 * width + 2] = 1;
  image[8 * width + 3] = 1;
  image[9 * width + 1] = 1;
  image[9 * width + 2] = 1;
  image[9 * width + 3] = 1;

  image[7 * width + 7] = 1;
  image[7 * width + 8] = 1;
  image[7 * width + 9] = 1;
  image[8 * width + 7] = 1;
  image[8 * width + 8] = 1;
  image[8 * width + 9] = 1;
  image[9 * width + 7] = 1;
  image[9 * width + 8] = 1;
  image[9 * width + 9] = 1;

  std::vector<int> hull(width * height, 0);

  std::vector<int> expected_hull = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_ten_components_image) {
  boost::mpi::communicator world;
  const int width = 20;
  const int height = 20;

  std::vector<int> image(width * height, 0);

  image[2 * width + 2] = 1;
  image[2 * width + 3] = 1;
  image[3 * width + 2] = 1;
  image[3 * width + 3] = 1;

  image[6 * width + 6] = 1;
  image[6 * width + 7] = 1;
  image[7 * width + 6] = 1;
  image[7 * width + 7] = 1;

  image[10 * width + 10] = 1;
  image[10 * width + 11] = 1;
  image[11 * width + 10] = 1;
  image[11 * width + 11] = 1;

  image[14 * width + 14] = 1;
  image[14 * width + 15] = 1;
  image[15 * width + 14] = 1;
  image[15 * width + 15] = 1;

  image[5 * width + 15] = 1;
  image[5 * width + 16] = 1;
  image[6 * width + 15] = 1;
  image[6 * width + 16] = 1;

  image[15 * width + 5] = 1;
  image[15 * width + 6] = 1;
  image[16 * width + 5] = 1;
  image[16 * width + 6] = 1;

  image[10 * width + 5] = 1;
  image[10 * width + 6] = 1;
  image[11 * width + 5] = 1;
  image[11 * width + 6] = 1;

  image[5 * width + 10] = 1;
  image[5 * width + 11] = 1;
  image[6 * width + 10] = 1;
  image[6 * width + 11] = 1;

  image[18 * width + 2] = 1;
  image[18 * width + 3] = 1;
  image[19 * width + 2] = 1;
  image[19 * width + 3] = 1;

  image[2 * width + 18] = 1;
  image[2 * width + 19] = 1;
  image[3 * width + 18] = 1;
  image[3 * width + 19] = 1;

  std::vector<int> hull(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI TestTaskMPI(taskDataPar);
  ASSERT_TRUE(TestTaskMPI.validation());
  TestTaskMPI.pre_processing();
  TestTaskMPI.run();
  TestTaskMPI.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_hull = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_different_width_and_height) {
  boost::mpi::communicator world;
  const int width = 6;
  const int height = 5;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> expected_hull = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                    1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI testTaskMPI(taskDataPar);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_four_corner_points) {
  boost::mpi::communicator world;
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height, 0);

  image[0 * width + 0] = 1;
  image[0 * width + 9] = 1;
  image[9 * width + 0] = 1;
  image[9 * width + 9] = 1;

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }
  chistov_a_convex_hull_image_mpi::ConvexHullMPI testTaskMPI(taskDataPar);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_power_of_two_size) {
  boost::mpi::communicator world;
  const int width = 128;
  const int height = 128;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI testTaskMPI(taskDataPar);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}

TEST(chistov_a_convex_hull_image_mpi, test_prime_numbers_size) {
  boost::mpi::communicator world;
  const int width = 343;
  const int height = 343;

  std::vector<int> image(width * height, 1);
  std::vector<int> hull(width * height);

  std::vector<int> expected_hull(width * height, 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected_hull[y * width + x] = 1;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
    taskDataPar->inputs_count.emplace_back(width * height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
    taskDataPar->outputs_count.emplace_back(width * height);
  }

  chistov_a_convex_hull_image_mpi::ConvexHullMPI testTaskMPI(taskDataPar);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_hull, hull);
  }
}