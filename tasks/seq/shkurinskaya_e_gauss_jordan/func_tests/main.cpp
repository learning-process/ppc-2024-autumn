#include <gtest/gtest.h>

#include "seq/shkurinskaya_e_gauss_jordan/include/ops_seq.hpp"

TEST(shkurinskaya_e_gauss_jordan_seq, Test_2x2) {
  int n = 2;
  std::vector<double> in = {2, 3, 5, 4, 1, 6};
  std::vector<double> expected_output = {1.3, 0.8};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_TRUE(gaussTaskSequential.validation());
  gaussTaskSequential.pre_processing();
  gaussTaskSequential.run();
  gaussTaskSequential.post_processing();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(shkurinskaya_e_gauss_jordan_seq, Test_Gauss_3x3) {
  int n = 3;

  std::vector<double> in = {1, 1, 1, 6, 0, 2, 5, -4, 2, 5, -1, 27};
  std::vector<double> expected_output = {5.0, 3.0, -2.0};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_EQ(gaussTaskSequential.validation(), true);
  gaussTaskSequential.pre_processing();
  gaussTaskSequential.run();
  gaussTaskSequential.post_processing();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(shkurinskaya_e_gauss_jordan_seq, Test_Gauss_5x5) {
  int n = 5;

  std::vector<double> in = {2, 3,  -1, 5, 1, 8, 4, -2, 3, -1, 2, 10, -1, 5, 2,
                            3, -4, -3, 3, 2, 4, 1, -2, 6, 1,  1, 1,  1,  1, 4};
  std::vector<double> expected_output = {2.0, 0.142, 0.285, 0.571, 1.0};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_TRUE(gaussTaskSequential.validation());
  gaussTaskSequential.pre_processing();
  gaussTaskSequential.run();
  gaussTaskSequential.post_processing();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(shkurinskaya_e_gauss_jordan_seq, Test_Gauss_Invalid_Data) {
  int n = 2;

  std::vector<double> in = {1, 2, 5};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_EQ(gaussTaskSequential.validation(), false);
}

TEST(shkurinskaya_e_gauss_jordan_seq, Test_Gauss_Zero_Diag) {
  int n = 3;
  std::vector<double> in = {0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 4, 3};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_FALSE(gaussTaskSequential.validation());
}

TEST(shkurinskaya_e_gauss_jordan_seq, Test_Gauss_Overdetermined_System) {
  int n = 3;

  std::vector<double> in = {1, 1, 1, 6, 0, 2, 5, -4, 2, 5, -1, 27, 1, 1, 1, 6};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size() / (n + 1));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shkurinskaya_e_gauss_jordan_seq::TestTaskSequential gaussTaskSequential(taskDataSeq);

  ASSERT_FALSE(gaussTaskSequential.validation());
}
