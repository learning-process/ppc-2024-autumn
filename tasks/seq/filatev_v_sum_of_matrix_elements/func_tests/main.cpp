// filatev_v_sum_of_matrix_elements_seq
#include <gtest/gtest.h>
#include <vector>
#include "seq/filatev_v_sum_of_matrix_elements/include/ops_seq.hpp"

TEST(filatev_v_sum_of_matrix_elements_seq, Test_Sum_10_10_1) {
  const int count = 10;

  // Create data
  std::vector<std::vector<int>> in(count, std::vector<int>(count, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < count; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  filatev_v_sum_of_matrix_elements_seq::SumMatrix sumMatrix(taskDataSeq);
  ASSERT_EQ(sumMatrix.validation(), true);
  sumMatrix.pre_processing();
  sumMatrix.run();
  sumMatrix.post_processing();

 
  ASSERT_EQ(100, out[0]);
}

TEST(filatev_v_sum_of_matrix_elements_seq, Test_Sum_10_20_1) {
  const int size_m = 10, size_n = 20;

  // Create data
  std::vector<std::vector<int>> in(size_m, std::vector<int>(size_n, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < size_m; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(size_n);
  taskDataSeq->inputs_count.emplace_back(size_m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  filatev_v_sum_of_matrix_elements_seq::SumMatrix sumMatrix(taskDataSeq);
  ASSERT_EQ(sumMatrix.validation(), true);
  sumMatrix.pre_processing();
  sumMatrix.run();
  sumMatrix.post_processing();

 
  ASSERT_EQ(200, out[0]);
}

TEST(filatev_v_sum_of_matrix_elements_seq, Test_Sum_20_10_1) {
  const int size_m = 20, size_n = 10;

  // Create data
  std::vector<std::vector<int>> in(size_m, std::vector<int>(size_n, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < size_m; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(size_n);
  taskDataSeq->inputs_count.emplace_back(size_m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  filatev_v_sum_of_matrix_elements_seq::SumMatrix sumMatrix(taskDataSeq);
  ASSERT_EQ(sumMatrix.validation(), true);
  sumMatrix.pre_processing();
  sumMatrix.run();
  sumMatrix.post_processing();

 
  ASSERT_EQ(200, out[0]);
}

TEST(filatev_v_sum_of_matrix_elements_seq, Test_Sum_1_1_1) {
  const int size_m = 1, size_n = 1;

  // Create data
  std::vector<std::vector<int>> in(size_m, std::vector<int>(size_n, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < size_m; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(size_n);
  taskDataSeq->inputs_count.emplace_back(size_m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  filatev_v_sum_of_matrix_elements_seq::SumMatrix sumMatrix(taskDataSeq);
  ASSERT_EQ(sumMatrix.validation(), true);
  sumMatrix.pre_processing();
  sumMatrix.run();
  sumMatrix.post_processing();

 
  ASSERT_EQ(1, out[0]);
}

TEST(filatev_v_sum_of_matrix_elements_seq, Test_Sum_10_20_different) {
  const int size_m = 10, size_n = 20;

  // Create data
  std::vector<std::vector<int>> in(size_m, std::vector<int>(size_n, 1));
  std::vector<int> out(1, 0);

  for (int i = 0; i < size_m; ++i){
    for (int j = 0; j < size_n; ++j){
      in[i][j] = (i * size_n + j + 1);
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < size_m; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(size_n);
  taskDataSeq->inputs_count.emplace_back(size_m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  filatev_v_sum_of_matrix_elements_seq::SumMatrix sumMatrix(taskDataSeq);
  ASSERT_EQ(sumMatrix.validation(), true);
  sumMatrix.pre_processing();
  sumMatrix.run();
  sumMatrix.post_processing();

  ASSERT_EQ(20100, out[0]);
}
