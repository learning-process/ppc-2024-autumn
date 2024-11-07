// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <limits>
#include <vector>

#include "seq/beresnev_a_min_values_by_matrix_columns/include/ops_seq.hpp"

TEST(beresnev_a_min_values_by_matrix_columns_seq, Empty_Input_0) {
  const int N = 0;
  const int M = 3;

  std::vector<int> in(N * M, 0);
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Empty_Input_1) {
  const int N = 6;
  const int M = 0;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Wrong_Size_0) {
  const int N = -2;
  const int M = 3;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Wrong_Size_1) {
  const int N = 2;
  const int M = 312;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Identity_Matrix) {
  const int N = 1;
  const int M = 1;

  std::vector<int> in{10};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  std::vector<int> gold{10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_0) {
  const int N = 2;
  const int M = 3;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  const std::vector<int> gold{-1, -100, 2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_1) {
  const int N = 100;
  const int M = 100;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (int i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 200 - 100;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < M; ++i) {
    int expectedMin = in[i];
    for (int j = 1; j < N; ++j) {
      int currentValue = in[j * M + i];
      if (currentValue < expectedMin) {
        expectedMin = currentValue;
      }
    }
    ASSERT_EQ(out[i], expectedMin);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_2) {
  const int N = 10000;
  const int M = 1;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (int i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 200 - 100;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  int expectedMin = in[0];
  for (int j = 1; j < N; ++j) {
    int currentValue = in[j * M];
    if (currentValue < expectedMin) {
      expectedMin = currentValue;
    }
  }
  ASSERT_EQ(out[0], expectedMin);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_3) {
  const int N = 1;
  const int M = 10000;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (int i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 200 - 100;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(in, out);
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_4) {
  const std::uint32_t N = 332;
  const std::uint32_t M = 875;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (std::uint32_t i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 2000 - 1000;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (std::uint32_t i = 0; i < M; ++i) {
    int expectedMin = in[i];
    for (std::uint32_t j = 1; j < N; ++j) {
      int currentValue = in[j * M + i];
      if (currentValue < expectedMin) {
        expectedMin = currentValue;
      }
    }
    ASSERT_EQ(out[i], expectedMin);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_seq, Test_Base_5) {
  const std::uint32_t N = 9271;
  const std::uint32_t M = 682;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (std::uint32_t i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 2000 - 1000;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (std::uint32_t i = 0; i < M; ++i) {
    int expectedMin = in[i];
    for (std::uint32_t j = 1; j < N; ++j) {
      int currentValue = in[j * M + i];
      if (currentValue < expectedMin) {
        expectedMin = currentValue;
      }
    }
    ASSERT_EQ(out[i], expectedMin);
  }
}