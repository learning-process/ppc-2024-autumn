#include <gtest/gtest.h>

#include <vector>
#include <random>

#include "seq/guseynov_e_marking_comps_of_bin_image/include/ops_seq.hpp"

std::vector<int> getRandomBinImage(int r, int c) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(0, 2);
  std::vector<int> vec(r * c);
  for (int i = 0; i < r * c; i++) {
    vec[i] = gen();
  }
  return vec;
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
  std::vector<int> out(rows * columns, -1);

  for (int i = 0; i < rows; i++){
    for (int j = 0; j < columns; j++){
      std::cout << in[i * columns] << " ";
    }
    std::cout << "\n";
  }

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
  std::cout << "\n";
    for (int i = 0; i < rows; i++){
    for (int j = 0; j < columns; j++){
      std::cout << out[i * columns] << " ";
    }
    std::cout << "\n";
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_one_column_with_isolated_point3) {
  const int rows = 10;
  const int columns = 10;
  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(rows * columns, -1);

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
  std::cout << "\n";
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < columns; j++){
      std::cout << out[i * columns] << " ";
    }
    std::cout << "\n";
  }

}
