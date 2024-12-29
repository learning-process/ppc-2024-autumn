#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/sozonov_i_image_filtering_vertical_gaussian_3x3/include/ops_seq.hpp"

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_image_less_than_3x3) {
  const int width = 2;
  const int height = 2;

  // Create data
  std::vector<double> in = {4, 6, 8, 24};
  std::vector<double> out(width * height, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_wrong_pixels) {
  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {143, 6, 853, -24, 31, -25, 1, 5, -7, 361, 28, 98, -45, 982, 461};
  std::vector<double> out(width * height, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_3x3) {
  const int width = 3;
  const int height = 3;

  // Create data
  std::vector<double> in = {4, 6, 8, 24, 31, 25, 1, 5, 7};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0, 0, 11.1875, 16.5, 12.6875, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_5x3) {
  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {34, 24, 27, 67, 42, 48, 93, 26, 47, 2, 34, 13, 81, 24, 32};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0, 0, 0, 0, 34.4375, 48.125, 45.5, 38, 21.3125, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_5x5) {
  const int width = 5;
  const int height = 5;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0,     0,    0,  0,  4,  6,  7, 8, 6.5, 7.75, 11, 12,
                             13, 10.25, 11.5, 16, 17, 18, 14, 0, 0, 0,   0,    0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_5x7) {
  const int width = 5;
  const int height = 7;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0,  0,     0,  0,  4,  6,     7,  8,  6.5, 7.75, 11,   12, 13, 10.25, 11.5, 16, 17,
                             18, 14, 15.25, 21, 22, 23, 17.75, 19, 26, 27,  28,   21.5, 0,  0,  0,     0,    0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_10x4) {
  const int width = 10;
  const int height = 4;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,     0,  0,  0,  0,  0,  0,  0,  0,  0,    7.75, 11, 12, 13, 14, 15, 16, 17, 18, 14,
                             15.25, 21, 22, 23, 24, 25, 26, 27, 28, 21.5, 0,    0,  0,  0,  0,  0,  0,  0,  0,  0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_100x100) {
  const int width = 100;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_150x100) {
  const int width = 150;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_image_filtering_vertical_gaussian_3x3_seq, test_120x200) {
  const int width = 120;
  const int height = 200;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}