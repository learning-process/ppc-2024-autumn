#include <gtest/gtest.h>

#include <vector>

#include "seq/vasilev_s_nearest_neighbor_elements/include/ops_seq.hpp"

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Small_Vector) {
  std::vector<int> input_vec = {5, 3, 8, 7, 2};
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  int expected_min_diff = 1;
  int expected_index1 = 2;
  int expected_index2 = 3;

  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Equal_Elements) {
  std::vector<int> input_vec = {7, 7, 7, 7, 7};
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  int expected_min_diff = 0;
  int expected_index1 = 0;
  int expected_index2 = 1;

  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Negative_Numbers) {
  std::vector<int> input_vec = {-10, -20, -15, -30, -25};
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  int expected_min_diff = 5;
  int expected_index1 = 1;
  int expected_index2 = 2;

  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Single_Element_Vector) {
  std::vector<int> input_vec = {42};
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Empty_Vector) {
  std::vector<int> input_vec;
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(vasilev_s_nearest_neighbor_elements_seq, Test_Large_Vector) {
  std::vector<int> input_vec = {100, 95, 90, 85, 80, 75, 70, 65, 60, 55};
  std::vector<int> output_result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasilev_s_nearest_neighbor_elements_seq::FindClosestNeighborsSequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  taskSequential.pre_processing();
  taskSequential.run();
  taskSequential.post_processing();

  int expected_min_diff = 5;
  int expected_index1 = 0;
  int expected_index2 = 1;

  ASSERT_EQ(output_result[0], expected_min_diff);
  ASSERT_EQ(output_result[1], expected_index1);
  ASSERT_EQ(output_result[2], expected_index2);
}
