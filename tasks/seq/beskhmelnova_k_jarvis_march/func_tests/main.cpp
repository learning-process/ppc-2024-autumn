#include <gtest/gtest.h>

#include <vector>

#include "seq/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

TEST(beskhmelnova_k_jarvis_march_seq, Test_with_2_points) {
  // Create data
  int num_points = 2;
  std::vector<double> x = {0.0, 1.0};
  std::vector<double> y = {0.0, 1.0};

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_empty_triangle) {
  // Create data
  int num_points = 3;
  std::vector<double> x = {0.0, 1.0, -1.0};
  std::vector<double> y = {1.0, -1.0, -1.0};

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 3;
  std::vector<double> res_x = {-1.0, 1.0, 0.0};
  std::vector<double> res_y = {-1.0, -1.0, 1.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_triangle_with_points) {
  // Create data
  int num_points = 7;
  std::vector<double> x = {0.0, 1.0, -1.0, 0.0, 0.0, -0.1, 0.11};
  std::vector<double> y = {1.0, -1.0, -1.0, 0.0, -0.5, -0.3, 0.11};

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 3;
  std::vector<double> res_x = {-1.0, 1.0, 0.0};
  std::vector<double> res_y = {-1.0, -1.0, 1.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_empty_square) {
  // Create data
  int num_points = 4;
  std::vector<double> x = {1.0, -1.0, -1.0, 1.0};
  std::vector<double> y = {1.0, 1.0, -1.0, -1.0};

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1.0, 1.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1.0, 1.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_square_with_points) {
  // Create data
  int num_points = 7;
  std::vector<double> x = {1.0, 0.0, -1.0, 0.35, -1.0, 1.0, 0.2};
  std::vector<double> y = {1.0, 0.0, 1.0, -0.7, -1.0, -1.0, 0.8};

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1.0, 1.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1.0, 1.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_square_with_20_random_points) {
  int num_points = 20;
  std::vector<double> x(num_points);
  std::vector<double> y(num_points);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 4; i < num_points; i++) {
    x[i] = std::rand() % 1000;
    y[i] = std::rand() % 1000;
  }
  x[0] = -1.0;
  y[0] = -1.0;

  x[1] = -1.0;
  y[1] = 1000.0;

  x[2] = 1000.0;
  y[2] = 1000.0;

  x[3] = 1000.0;
  y[3] = -1.0;

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}

TEST(beskhmelnova_k_jarvis_march_seq, Test_square_with_100_random_points) {
  int num_points = 100;
  std::vector<double> x(num_points);
  std::vector<double> y(num_points);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 4; i < num_points; i++) {
    x[i] = std::rand() % 1000;
    y[i] = std::rand() % 1000;
  }
  x[0] = -1.0;
  y[0] = -1.0;

  x[1] = -1.0;
  y[1] = 1000.0;

  x[2] = 1000.0;
  y[2] = 1000.0;

  x[3] = 1000.0;
  y[3] = -1.0;

  int hull_size;
  std::vector<double> hull_x(num_points, 0);
  std::vector<double> hull_y(num_points, 0);

  int res_size = 4;
  std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
  std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->inputs_count.emplace_back(x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
  taskDataSeq->inputs_count.emplace_back(y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
  taskDataSeq->outputs_count.emplace_back(hull_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
  taskDataSeq->outputs_count.emplace_back(hull_y.size());

  // Create Task
  beskhmelnova_k_jarvis_march_seq::TestTaskSequential<double> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(hull_size, res_size);

  for (int i = 0; i < res_size; i++) {
    ASSERT_EQ(hull_x[i], res_x[i]);
    ASSERT_EQ(hull_y[i], res_y[i]);
  }
}