#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/gromov_a_convex_hull/include/ops_seq.hpp"

namespace gromov_a_convex_hull_seq_test {
std::vector<int> createCanvas(int width, int height) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1);

  std::vector<int> grid(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      grid[y * width + x] = dist(gen);
    }
  }
  return grid;
}
}  // namespace gromov_a_convex_hull_seq_test

TEST(gromov_a_convex_hull_seq, Test_Small_Square_1) {
  const int width = 10;
  const int height = 10;

  std::vector<int> grid(width * height, 0);
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(grid.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  gromov_a_convex_hull_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(grid, hull);
}

TEST(gromov_a_convex_hull_seq, Test_Empty_Hull_Large_Image) {
  const int width = 100;
  const int height = 100;

  std::vector<int> grid(width * height, 0);
  std::vector<int> hull(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(grid.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  gromov_a_convex_hull_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_FALSE(hull.empty());
}

TEST(gromov_a_convex_hull_seq, Test_Empty) {
  const int width = 0;
  const int height = 0;

  std::vector<int> grid;
  std::vector<int> hull;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(grid.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  gromov_a_convex_hull_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(gromov_a_convex_hull_seq, Test_Small_Square_2) {
  const int width = 3;
  const int height = 3;

  std::vector<int> grid = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> hull(width * height, 0);
  std::vector<int> expected_hull = {1, 1, 1, 1, 0, 1, 1, 1, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(grid.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  gromov_a_convex_hull_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(hull, expected_hull);
}

TEST(gromov_a_convex_hull_seq, Test_Some_Image) {
  const int width = 10;
  const int height = 10;

  std::vector<int> grid(width * height, 0);
  std::vector<int> hull(width * height, 0);
  std::vector<int> expected_hull(width * height, 0);
  grid[5 * width + 5] = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(grid.data()));
  taskDataSeq->inputs_count.emplace_back(width * height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull.data()));
  taskDataSeq->outputs_count.emplace_back(width * height);

  gromov_a_convex_hull_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_hull, hull);
}
