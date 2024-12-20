#include <gtest/gtest.h>

#include "seq/kudryashova_i_graham's_scan/include/Graham'sScanSeq.hpp"

void generateUniquePoints(int numPoints, int8_t minX, int8_t maxX, int8_t minY, int8_t maxY,
                          std::vector<int8_t> &xCoords, std::vector<int8_t> &yCoords) {
  if (numPoints > (maxX - minX + 1) * (maxY - minY + 1)) {
    std::cerr << "Error: Not enough unique points can be generated in the given range." << std::endl;
    return;
  }
  std::vector<std::pair<int8_t, int8_t>> allPoints;
  for (int8_t x = minX; x <= maxX; x += 1) {
    for (int8_t y = minY; y <= maxY; y += 1) {
      allPoints.emplace_back(x, y);
    }
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(allPoints.begin(), allPoints.end(), gen);
  for (int i = 0; i < numPoints; ++i) {
    xCoords.push_back(allPoints[i].first);
    yCoords.push_back(allPoints[i].second);
  }
}

void addAns(std::vector<int8_t> &v1, std::vector<int8_t> &v2, int value) {
  v1.push_back(-value);
  v2.push_back(-value);
  v1.push_back(value);
  v2.push_back(-value);
  v1.push_back(value);
  v2.push_back(value);
  v1.push_back(-value);
  v2.push_back(value);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_test_square) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x = {0, 0, 1, 1};
  std::vector<int8_t> vector_y = {1, 0, 0, 1};
  std::vector<int8_t> result(8, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int8_t> gold{0, 0, 1, 0, 1, 1, 0, 1};
  ASSERT_EQ(result, gold);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_test_star) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x = {0, 1, 4, 2, 3, 0, -3, -2, -4, -1};
  std::vector<int8_t> vector_y = {4, 1, 1, -1, -4, -2, -4, -1, 1, 1};
  std::vector<int8_t> result(10, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int8_t> gold{-3, -4, 3, -4, 4, 1, 0, 4, -4, 1};
  ASSERT_EQ(result, gold);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_simple_test_1) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x = {3, 0, 2, 3, 1, -1, 1, 0, 3, -3, -3};
  std::vector<int8_t> vector_y = {5, 3, 2, 2, 1, 1, 0, 0, -1, 2, -2};
  std::vector<int8_t> result(8, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int8_t> gold{-3, -2, 3, -1, 3, 5, -3, 2};
  ASSERT_EQ(result, gold);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_simple_test_2) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x = {5, 3, 3, 1, 2, 4, 1, 1, 2, 1, -1, -2, -1, -1};
  std::vector<int8_t> vector_y = {3, 3, 2, 2, 1, -2, -1, -2, -3, -4, -3, -1, 1, 3};
  std::vector<int8_t> result(12, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int8_t> gold{1, -4, 4, -2, 5, 3, -1, 3, -2, -1, -1, -3};
  ASSERT_EQ(result, gold);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_simple_random_test) {
  const int count_size = 15;
  const int ans_number = 10;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x;
  std::vector<int8_t> vector_y;
  generateUniquePoints(count_size, -(ans_number - 1), (ans_number - 1), -(ans_number - 1), (ans_number - 1), vector_x,
                       vector_y);
  addAns(vector_x, vector_y, ans_number);
  std::vector<int8_t> result(8, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int8_t> answer = {-ans_number, -ans_number, ans_number,  -ans_number,
                                ans_number,  ans_number,  -ans_number, ans_number};
  ASSERT_EQ(result, answer);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_check_same_number_x_and_y) {
  const int count = 8;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(count, 0);
  std::vector<int8_t> vector_x = {2, 0, -2, 0};
  std::vector<int8_t> vector_y = {0, 2, 0, -2};
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_check_not_same_number_x_and_y) {
  const int count = 8;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> result(count, 0);
  std::vector<int8_t> vector_x = {2, 0, -2, 0, 2};
  std::vector<int8_t> vector_y = {0, 2, 0, -2};
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_check_empty_vertex) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x;
  std::vector<int8_t> vector_y;
  std::vector<int8_t> result(8, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kudryashova_i_graham_scan_seq, seq_graham_scan_test_1_vertex) {
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x = {1};
  std::vector<int8_t> vector_y = {1};
  std::vector<int8_t> result(3, 0);
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  kudryashova_i_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}