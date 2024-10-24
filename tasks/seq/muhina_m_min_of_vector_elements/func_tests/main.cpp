// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/muhina_m_min_of_vector_elements/include/ops_seq.hpp"

TEST(muhina_m_min_of_vector_elements_seq, Test_Min_10) {
  const int count = 10;

  // Create data
  std::vector<int> in(count, 100);
  in[1] = 0;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  muhina_m_min_of_vector_elements_seq::MinOfVectorSequential MinOfVectorSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorSequential.validation(), true);
  MinOfVectorSequential.pre_processing();
  MinOfVectorSequential.run();
  MinOfVectorSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(muhina_m_min_of_vector_elements_seq, Test_Min_20) {
  const int count = 20;

  // Create data
  std::vector<int> in(count, 100);
  in[1] = 0;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  muhina_m_min_of_vector_elements_seq::MinOfVectorSequential MinOfVectorSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorSequential.validation(), true);
  MinOfVectorSequential.pre_processing();
  MinOfVectorSequential.run();
  MinOfVectorSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}


TEST(muhina_m_min_of_vector_elements_seq, Test_Min_50) {
  const int count = 50;

  // Create data
  std::vector<int> in(count, 100);
  in[1] = 0;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  muhina_m_min_of_vector_elements_seq::MinOfVectorSequential MinOfVectorSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorSequential.validation(), true);
  MinOfVectorSequential.pre_processing();
  MinOfVectorSequential.run();
  MinOfVectorSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(muhina_m_min_of_vector_elements_seq, Test_Min_70) {
  const int count = 70;

  // Create data
  std::vector<int> in(count, 100);
  in[1] = 0;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  muhina_m_min_of_vector_elements_seq::MinOfVectorSequential MinOfVectorSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorSequential.validation(), true);
  MinOfVectorSequential.pre_processing();
  MinOfVectorSequential.run();
  MinOfVectorSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(muhina_m_min_of_vector_elements_seq, Test_Min_100) {
  const int count = 100;

  // Create data
  std::vector<int> in(count, 100);
  in[1] = 0;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  muhina_m_min_of_vector_elements_seq::MinOfVectorSequential MinOfVectorSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorSequential.validation(), true);
  MinOfVectorSequential.pre_processing();
  MinOfVectorSequential.run();
  MinOfVectorSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}
