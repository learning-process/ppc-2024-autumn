// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "seq/beresnev_a_cannons_algorithm/include/ops_seq.hpp"

static std::vector<double> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = gen();
  }
  return vec;
}

static std::vector<double> getAns(std::vector<double> &A, std::vector<double> &B, size_t N) {
  std::vector<double> C(N * N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
  return C;
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Empty_In) {
  size_t n = 1;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB;
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size() + 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Empty_Out) {
  size_t n = 1;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Wrong_Size) {
  size_t n = 10;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size() + 1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Wrong_Size_1) {
  size_t n = 10;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size() - 1);

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_cannons_algorithm_seq, Test_m1_1) {
  size_t n = 1;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);
  std::vector<double> ans = getAns(inA, inB, n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(ans.begin(), ans.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Inverse) {
  size_t n = 3;
  std::vector<double> iden{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> inA{2, 1, 1, 1, 3, 2, 1, 0, 0};
  std::vector<double> inB{0, 0, 1, -2, 1, 3, 3, -1, -5};

  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(iden.begin(), iden.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Iden) {
  size_t n = 3;
  std::vector<double> inA{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(inB.begin(), inB.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Iden_1) {
  size_t n = 3;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Random) {
  size_t n = 13;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);
  std::vector<double> ans = getAns(inA, inB, n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(ans.begin(), ans.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, Test_Random_1) {
  size_t n = 199;
  std::vector<double> inA = getRandomVector(n * n);
  std::vector<double> inB = getRandomVector(n * n);
  std::vector<double> outC(n * n);
  std::vector<double> ans = getAns(inA, inB, n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  beresnev_a_cannons_algorithm_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(
      std::equal(ans.begin(), ans.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}