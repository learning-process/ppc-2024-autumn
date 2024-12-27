#include <gtest/gtest.h>

#include <vector>

#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

TEST(kurakin_m_graham_scan_seq, Test_shell_rhomb) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 4;
  points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = 4;
  std::vector<double> ans = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    ASSERT_EQ(scan_points[i], ans[i]);
    ASSERT_EQ(scan_points[i + 1], ans[i + 1]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_square) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 4;
  points = {2.0, 2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = 4;
  std::vector<double> ans = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    ASSERT_EQ(scan_points[i], ans[i]);
    ASSERT_EQ(scan_points[i + 1], ans[i + 1]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_rhomb_with_inside_points) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 17;
  points = {0.3, -0.25, 1.0, 0.0,   2.0, 0.0,  0.3, 0.25, 0.0,  -2.0, 0.0, -1.0, 0.25, -0.3, -0.25, -0.3, 0.0,
            1.0, 0.0,   2.0, -0.25, 0.3, 0.25, 0.3, -0.3, 0.25, -1.0, 0.0, -2.0, 0.0,  -0.3, -0.25, 0.1,  0.1};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = 4;
  std::vector<double> ans = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    ASSERT_EQ(scan_points[i], ans[i]);
    ASSERT_EQ(scan_points[i + 1], ans[i + 1]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_square_with_inside_points) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 17;
  points = {-2.0, -2.0, -1.0, -1.0, -0.5, -1.0, -1.0, -0.5, 2.0, -2.0, 0.5, -1.0, 1.0, -1.0, 1.0, -0.5, 2.0,
            2.0,  1.0,  1.0,  0.5,  1.0,  1.0,  0.5,  -2.0, 2.0, -0.5, 1.0, -1.0, 1.0, -1.0, 0.5, 0.1,  0.1};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = 4;
  std::vector<double> ans = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    ASSERT_EQ(scan_points[i], ans[i]);
    ASSERT_EQ(scan_points[i + 1], ans[i + 1]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_validation_count_points) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 2;
  points = {2.0, 2.0, 1.0, 1.0};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_inputs_point) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 4;
  points = {2.0, 2.0, 1.0, 1.0, -2.0, 2.0, 1.0};

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_outputs_point) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 4;
  points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

  int scan_size;
  std::vector<double> scan_points(count_point * 2 - 1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_inputs_count) {
  int count_point;
  std::vector<double> points;
  // Create data
  count_point = 4;

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_outputs_count) {
  std::vector<double> points;
  // Create data
  points = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};

  int scan_size;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}
