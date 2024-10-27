#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/solovyev_d_vector_max/include/header.hpp"

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

TEST(solovyev_d_vector_max_mpi, Test_Max_10) {
  const int count = 10;

  // Create data
  std::vector<int> in = getRandomVector(count);
  in[count / 2] = 1024;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxSequential(taskDataSeq);
  ASSERT_EQ(VectorMaxSequential.validation(), true);
  VectorMaxSequential.pre_processing();
  VectorMaxSequential.run();
  VectorMaxSequential.post_processing();
  ASSERT_EQ(1024, out[0]);
}

TEST(solovyev_d_vector_max_mpi, Test_Max_100) {
  const int count = 20;

  // Create data
  std::vector<int> in = getRandomVector(count);
  in[count / 2] = 1024;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxSequential(taskDataSeq);
  ASSERT_EQ(VectorMaxSequential.validation(), true);
  VectorMaxSequential.pre_processing();
  VectorMaxSequential.run();
  VectorMaxSequential.post_processing();
  ASSERT_EQ(1024, out[0]);
}

TEST(solovyev_d_vector_max_mpi, Test_Max_1000) {
  const int count = 50;

  // Create data
  std::vector<int> in = getRandomVector(count);
  in[count / 2] = 1024;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxSequential(taskDataSeq);
  ASSERT_EQ(VectorMaxSequential.validation(), true);
  VectorMaxSequential.pre_processing();
  VectorMaxSequential.run();
  VectorMaxSequential.post_processing();
  ASSERT_EQ(1024, out[0]);
}

TEST(solovyev_d_vector_max_mpi, Test_Max_10000) {
  const int count = 70;

  // Create data
  std::vector<int> in = getRandomVector(count);
  in[count / 2] = 1024;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxSequential(taskDataSeq);
  ASSERT_EQ(VectorMaxSequential.validation(), true);
  VectorMaxSequential.pre_processing();
  VectorMaxSequential.run();
  VectorMaxSequential.post_processing();
  ASSERT_EQ(1024, out[0]);
}

TEST(solovyev_d_vector_max_mpi, Test_Max_100000) {
  const int count = 100;

  // Create data
  std::vector<int> in = getRandomVector(count);
  in[count / 2] = 1024;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_vector_max_mpi::VectorMaxSequential VectorMaxSequential(taskDataSeq);
  ASSERT_EQ(VectorMaxSequential.validation(), true);
  VectorMaxSequential.pre_processing();
  VectorMaxSequential.run();
  VectorMaxSequential.post_processing();
  ASSERT_EQ(1024, out[0]);
}
