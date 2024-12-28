#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "seq/sidorina_p_convex_hull_binary_image_seq/include/ops_seq.hpp"

namespace sidorina_p_convex_hull_binary_image_seq {
std::vector<int> gen(int width, int height) {
  std::vector<int> image(width * height);
  for (int i = 0; i < width * height; ++i) {
    image[i] = rand() % 2;
  }

  return image;
}
}  // namespace sidorina_p_convex_hull_binary_image_seq

using Params = std::tuple<int, int, std::vector<int>, std::vector<int>>;

class sidorina_p_convex_hull_binary_image_seq_test : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(sidorina_p_convex_hull_binary_image_seq_test, Test_image) {
  const auto &[width, height, image, ref] = GetParam();
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(image.data())));
  taskDataSeq->inputs_count.emplace_back(image.size());
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

INSTANTIATE_TEST_SUITE_P(
    sidorina_p_convex_hull_binary_image_seq_test_val, sidorina_p_convex_hull_binary_image_seq_test,
    ::testing::Values(Params(3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      Params(3, 3, {0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      Params(3, 3, {0, 1, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 1, 0, 1, 1, 0, 1, 1}),
                      Params(3, 4, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0}),
                      Params(4, 4, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0},
                             {1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0}),
                      Params(5, 5, {0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0},
                             {0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0})));

using Params_val = std::tuple<int, int, std::vector<int>>;

class sidorina_p_convex_hull_binary_image_seq_test_val : public ::testing::TestWithParam<Params_val> {
 protected:
};

TEST_P(sidorina_p_convex_hull_binary_image_seq_test_val, Test_validation) {
  int width = std::get<0>(GetParam());
  int height = std::get<1>(GetParam());
  std::vector<int> image = std::get<2>(GetParam());
  std::vector<int> hull(width * height);
  std::vector<int> ref(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  taskDataSeq->inputs_count.emplace_back(image.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  sidorina_p_convex_hull_binary_image_seq::ConvexHullBinImgSeq TestTaskSequential(taskDataSeq);
  ASSERT_FALSE(TestTaskSequential.validation());
}

INSTANTIATE_TEST_SUITE_P(sidorina_p_convex_hull_binary_image_seq_test_val,
                         sidorina_p_convex_hull_binary_image_seq_test_val,
                         ::testing::Values(Params_val(0, 6, {1}), Params_val(3, 0, {1}), Params_val(5, 5, {2}),
                                           Params_val(5, 5, {})));