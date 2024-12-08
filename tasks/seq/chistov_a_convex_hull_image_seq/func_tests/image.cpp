#include "seq/chistov_a_convex_hull_image_seq/include/image.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace chistov_a_convex_hull_image_seq_test {
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
}  // namespace chistov_a_convex_hull_image_seq_test

TEST(chistov_a_convex_hull_image_seq, validation_test_empty_vector) {
  const int width = 3;
  const int height = 4;
  std::vector<int> image;
  std::vector<int> hull;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(chistov_a_convex_hull_image_seq, validation_test_zero_height_and_width) {
  const int width = 0;
  const int height = 0;
  std::vector<int> image = chistov_a_convex_hull_image_seq_test::generateImage(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(chistov_a_convex_hull_image_seq, validation_test_negative_size) {
  const int width = -1;
  const int height = -1;
  std::vector<int> image = chistov_a_convex_hull_image_seq_test::generateImage(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(chistov_a_convex_hull_image_seq, validation_test_empty_output) {
  const int width = 10;
  const int height = 10;
  std::vector<int> image = chistov_a_convex_hull_image_seq_test::generateImage(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(chistov_a_convex_hull_image_seq, validation_not_binary_image) {
  const int width = 10;
  const int height = 10;
  std::vector<int> image = chistov_a_convex_hull_image_seq_test::generateImage(width, height);
  image[0] = 5;
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width * height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(chistov_a_convex_hull_image_seq, test_image_of_zeros) {
  const int width = 10;
  const int height = 10;

  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  ASSERT_EQ(image, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_single_points_image) {
  const int width = 10;
  const int height = 10;
  std::vector<int> image(width * height, 0);
  std::vector<int> excepted_hull(width * height, 0);
  image[5 * width + 5] = 1;

  std::vector<int> hull(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(excepted_hull, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_non_adjacent_points) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(excepted_hull, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_one_component_image) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(excepted_hull, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_two_components_image) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  ASSERT_EQ(expected_hull, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_three_components_image) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ TestTaskSequential(taskDataSeq);
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(expected_hull, hull);
}

TEST(chistov_a_convex_hull_image_seq, test_four_corner_points) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ testTaskMPI(taskDataSeq);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  ASSERT_EQ(hull, expected_hull);
}

TEST(chistov_a_convex_hull_image_seq, test_different_width_and_height) {
  const int width = 6;
  const int height = 5;

  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height, 0);

  image[1 * width + 1] = 1;
  image[1 * width + 1] = 1;
  image[2 * width + 2] = 1;
  image[2 * width + 2] = 1;
  image[3 * width + 1] = 1;
  image[3 * width + 1] = 1;
  image[3 * width + 2] = 1;
  image[3 * width + 2] = 1;

  std::vector<int> expected_hull = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                                    1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ testTaskMPI(taskDataSeq);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  ASSERT_EQ(hull, expected_hull);
}

TEST(chistov_a_convex_hull_image_seq, test_power_of_two_size) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ testTaskMPI(taskDataSeq);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  ASSERT_EQ(hull, expected_hull);
}

TEST(chistov_a_convex_hull_image_seq, test_prime_numbers_size) {
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

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  chistov_a_convex_hull_image_seq::ConvexHullSEQ testTaskMPI(taskDataSeq);

  ASSERT_TRUE(testTaskMPI.validation());
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();

  ASSERT_EQ(hull, expected_hull);
}