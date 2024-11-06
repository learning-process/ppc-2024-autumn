
#include <gtest/gtest.h>

#include <vector>

#include "seq/laganina_e_sum_values_by_columns_matrix/include/ops_seq.hpp"

TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_2_2_matrix) {
  int n = 2;
  int m = 2;

  // Create data 555
  std::vector<int> in = {1, 2, 1, 2};
  std::vector<int> emp(m, 0);
  std::vector<int> out = {2, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(emp.data()));
  taskDataSeq->outputs_count.emplace_back(emp.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(emp, out);
}

TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_500_500_matrix) {
  // Create data

  int n = 500;
  int m = 500;
  std::vector<int> in(m * n, 0);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
  taskDataSeq->outputs_count.emplace_back(empty.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, empty);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_1000_1000_matrix) {
  // Create data

  int n = 1000;
  int m = 1000;
  std::vector<int> in(m * n, 0);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
  taskDataSeq->outputs_count.emplace_back(empty.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, empty);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_2000_2000_matrix) {
  // Create data

  int n = 2000;
  int m = 2000;
  std::vector<int> in(m * n, 0);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
  taskDataSeq->outputs_count.emplace_back(empty.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, empty);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_2_3_matrix) {
  // Create data

  int n = 3;
  int m = 2;
  std::vector<int> in(m * n, 1);
  std::vector<int> emp(n, 0);
  std::vector<int> out(n, m);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(emp.data()));
  taskDataSeq->outputs_count.emplace_back(emp.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, emp);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_3_2_matrix) {
  // Create data

  int n = 2;
  int m = 3;
  std::vector<int> in(m * n, 1);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, m);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
  taskDataSeq->outputs_count.emplace_back(empty.size());

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, empty);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_validation_output) {
  // Create data
  std::vector<int> in = {1, 2, 1, 2};
  int n = 2;
  int m = 2;
  std::vector<int> empty(n, 0);
  std::vector<int> out = {2, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n - 1);

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_NE(testTaskSequential.validation(), true);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_validation_empty) {
  // Create data
  std::vector<int> in = {};
  int n = 0;
  int m = 0;
  std::vector<int> empty = {};
  std::vector<int> out = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_NE(testTaskSequential.validation(), true);
}
TEST(laganina_e_sum_values_by_columns_matrix_seq, Test_validation_rank) {
  // Create data
  std::vector<int> in = {1, 2, 3};
  int n = 2;
  int m = 2;
  std::vector<int> empty(n, 0);
  std::vector<int> out = {4, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq testTaskSequential(taskDataSeq);
  ASSERT_NE(testTaskSequential.validation(), true);
}
