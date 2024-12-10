#include <gtest/gtest.h>

#include <vector>

#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

TEST(komshina_d_min_of_vector_elements_seq, Test_Min_1) {
  const int count = 50000;
  const int start_value = 1000000;
  const int decrement = 10;
  const int expected_min = start_value - decrement * (count - 1);

  std::vector<int> in(count);
  for (int i = 0; i < count; ++i) {
    in[i] = start_value - i * decrement;
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), true);
  MinOfVectorElementTaskSequential.pre_processing();
  MinOfVectorElementTaskSequential.run();
  MinOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(expected_min, out[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Test_Min_2) {
  const int count = 5000000;
  const int start_value = -1;
  const int decrement = 10;
  const int expected_min = start_value - decrement * (count - 1);

  std::vector<int> in(count);
  for (int i = 0; i < count; ++i) {
    in[i] = start_value - i * decrement;
  }
  std::cout << expected_min;
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), true);
  MinOfVectorElementTaskSequential.pre_processing();
  MinOfVectorElementTaskSequential.run();
  MinOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(expected_min, out[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Test_Min_3) {
  const int count = 10;
  const int expected_min = INT_MIN;

  std::vector<int> in(count, 0);
  std::vector<int> out(1);
  for (int i = 1; i < count; i += 1) {
    in[i] = INT_MIN;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), true);
  MinOfVectorElementTaskSequential.pre_processing();
  MinOfVectorElementTaskSequential.run();
  MinOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(expected_min, out[0]);
}

TEST(komshina_d_min_of_vector_elements_seq, Validation_InvalidOutputCount) {
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(0, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);

  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), false);
}

TEST(komshina_d_min_of_vector_elements_seq, Empty_Vector) {
  std::vector<int> in;
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), false);
}

TEST(komshina_d_min_of_vector_elements_seq, All_Elements_Equal) {
  const int count = 1000;
  const int value = 100;
  const int expected_min = value;

  std::vector<int> in(count, value);
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential MinOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(MinOfVectorElementTaskSequential.validation(), true);
  MinOfVectorElementTaskSequential.pre_processing();
  MinOfVectorElementTaskSequential.run();
  MinOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(expected_min, out[0]);
}