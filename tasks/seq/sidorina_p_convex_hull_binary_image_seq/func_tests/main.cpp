#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "seq/sidorina_p_convex_hull_binary_image_seq/include/ops_seq.hpp"

std::vector<int> gen(int width, int height) {
  if (width <= 0 || height <= 0) {
    return {};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1);

  std::vector<int> image(width * height);

  for (int i = 0; i < width * height; ++i) {
    image[i] = dist(gen) ? 1 : 0;
  }

  return image;
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_valid_not_bin) {
  const int width = 2;
  const int height = 4;
  std::vector<int> image = gen(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  image[2] = 2;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width * height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_valid_width_0) {
  const int width = 0;
  const int height = 4;
  std::vector<int> image = gen(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_valid_neg) {
  const int width = -2;
  const int height = -4;
  std::vector<int> image = gen(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_valid_height_0) {
  const int width = 2;
  const int height = 0;
  std::vector<int> image = gen(width, height);
  std::vector<int> hull(width * height);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_valid_vect_0) {
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

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_all_px_0) {
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

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  ASSERT_EQ(image, hull);
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_one_px_1) {
  const int width = 10;
  const int height = 10;
  std::vector<int> image(width * height, 0);
  std::vector<int> ref(width * height, 0);
  std::vector<int> hull(width * height, 0);

  image[1 * width + 2] = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  ASSERT_EQ(ref, hull);
}

TEST(sidorina_p_convex_hull_binary_image_seq, Test_image) {
  const int width = 5;
  const int height = 5;

  std::vector<int> image(width * height, 0);
  std::vector<int> hull(width * height, 0);

  image[0] = 1;
  image[1 * width + 4] = 1;
  image[3 * width + 3] = 1;

  std::vector<int> ref = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(ref, hull);
}
