
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "seq/komshina_d_min_of_vector_elements/include/ops_seq.hpp"

TEST(Sequential, Test_Min_1) {
  const int count = 10000;
  const int last = 0;

  std::vector<int> in(count);
  std::vector<int> out(1);
  for (int i = 0; i < count; ++i) {
    in[i] = i;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
  minOfVectorElementTaskSequential.pre_processing();
  minOfVectorElementTaskSequential.run();
  minOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(last, out[0]);
}

TEST(Sequential, Test_Min_2) {
  const int count = 10000;      
  const int min = -10; 

  std::vector<int> in(count);
  std::vector<int> out(1);

  for (int i = 0; i < count - 1; ++i) {
    in[i] = i;  
  }

  for (int i = count - 1; i < count; ++i) {
    in[i] = min; 
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
  minOfVectorElementTaskSequential.pre_processing();
  minOfVectorElementTaskSequential.run();
  minOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(min, out[0]);
}

TEST(Sequential, Test_Min_3) {
  const int count = 10;
  const int start = 500;
  const int min = -10;

  std::vector<int> in(count, start);

  std::random_device dev;
  std::mt19937 gen(dev());

  for (int i = 0; i < count - 1; i++) {
    in[i] = gen() % 1000;
  }

  in[count - 10] = min;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
  minOfVectorElementTaskSequential.pre_processing();
  minOfVectorElementTaskSequential.run();
  minOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(min, out[0]);
}

TEST(Sequential, Test_Min_4) {
  const int count = 1000;
  const int start = 200;
  const int min = 10;

  std::vector<int> in(count, start);

  std::random_device dev;
  std::mt19937 gen(dev());


  for (int i = 0; i < count - 1; i++) {
    in[i] = 100 + (gen() % 900); 
  }


  in[count - 10] = min;

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  komshina_d_min_of_vector_elements_seq::MinOfVectorElementTaskSequential minOfVectorElementTaskSequential(taskDataSeq);
  ASSERT_EQ(minOfVectorElementTaskSequential.validation(), true);
  minOfVectorElementTaskSequential.pre_processing();
  minOfVectorElementTaskSequential.run();
  minOfVectorElementTaskSequential.post_processing();
  ASSERT_EQ(min, out[0]);
}


