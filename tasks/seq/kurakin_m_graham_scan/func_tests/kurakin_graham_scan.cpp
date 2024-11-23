#include <gtest/gtest.h>

#include <vector>

#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

TEST(kurakin_m_graham_scan_seq, Test_shell_rhomb) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, 0.0, -2.0, 0.0};
  point_y = {0.0, 2.0, 0.0, -2.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = 4;
  std::vector<double> ans_x = {0.0, 2.0, 0.0, -2.0};
  std::vector<double> ans_y = {-2.0, 0.0, 2.0, 0.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size; i++) {
    ASSERT_EQ(scan_x[i], ans_x[i]);
    ASSERT_EQ(scan_y[i], ans_y[i]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_square) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, -2.0, -2.0, 2.0};
  point_y = {2.0, 2.0, -2.0, -2.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = 4;
  std::vector<double> ans_x = {2.0, 2.0, -2.0, -2.0};
  std::vector<double> ans_y = {-2.0, 2.0, 2.0, -2.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size; i++) {
    ASSERT_EQ(scan_x[i], ans_x[i]);
    ASSERT_EQ(scan_y[i], ans_y[i]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_rhomb_with_inside_points) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 17;
  point_x = {0.3, 1.0, 2.0, 0.3, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, -0.25, 0.25, -0.3, -1.0, -2.0, -0.3, 0.1};
  point_y = {-0.25, 0.0, 0.0, 0.25, -2.0, -1.0, -0.3, -0.3, 1.0, 2.0, 0.3, 0.3, 0.25, 0.0, 0.0, -0.25, 0.1};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = 4;
  std::vector<double> ans_x = {0.0, 2.0, 0.0, -2.0};
  std::vector<double> ans_y = {-2.0, 0.0, 2.0, 0.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size; i++) {
    ASSERT_EQ(scan_x[i], ans_x[i]);
    ASSERT_EQ(scan_y[i], ans_y[i]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_shell_square_with_inside_points) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 17;
  point_x = {-2.0, -1.0, -0.5, -1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, -2.0, -0.5, -1.0, -1.0, 0.1};
  point_y = {-2.0, -1.0, -1.0, -0.5, -2.0, -1.0, -1.0, -0.5, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 0.1};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = 4;
  std::vector<double> ans_x = {2.0, 2.0, -2.0, -2.0};
  std::vector<double> ans_y = {-2.0, 2.0, 2.0, -2.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(scan_size, ans_size);
  for (int i = 0; i < ans_size; i++) {
    ASSERT_EQ(scan_x[i], ans_x[i]);
    ASSERT_EQ(scan_y[i], ans_y[i]);
  }
}

TEST(kurakin_m_graham_scan_seq, Test_validation_count_points) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 2;
  point_x = {2.0, 1.0};
  point_y = {2.0, 1.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_inputs_point) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, 1.0, -2.0};
  point_y = {2.0, 1.0, 2.0, 1.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_outputs_point) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, 1.0, -2.0, -1.0};
  point_y = {2.0, 1.0, 2.0, 1.0};

  int scan_size;
  std::vector<double> scan_x(count_point - 1, 0);
  std::vector<double> scan_y(count_point, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_inputs_count) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, 1.0, -2.0, -1.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(kurakin_m_graham_scan_seq, Test_validation_outputs_count) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;
  // Create data
  count_point = 4;
  point_x = {2.0, 1.0, -2.0, -1.0};
  point_x = {2.0, 1.0, -2.0, -1.0};

  int scan_size;
  std::vector<double> scan_x(count_point, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());

  // Create Task
  kurakin_m_graham_scan_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}
