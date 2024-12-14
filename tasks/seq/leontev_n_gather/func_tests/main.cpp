// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/leontev_n_gather/include/ops_seq.hpp"

template <class InOutType>
void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_mat, std::vector<InOutType> &global_res) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataPar->inputs_count.emplace_back(global_mat.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataPar->outputs_count.emplace_back(global_res.size());
}

TEST(leontev_n_mat_vec_seq, 5x5_mat_vec) {
  // Create data
  std::vector<int32_t> invec(5, 10);
  std::vector<int32_t> inmat(25, 1);
  const std::vector<int32_t> expected_vec(5, 50);
  std::vector<int32_t> out(5, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, invec, inmat, out);
  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<int32_t> matVecSequential(taskDataSeq);
  ASSERT_TRUE(matVecSequential.validation());
  matVecSequential.pre_processing();
  matVecSequential.run();
  matVecSequential.post_processing();
  ASSERT_EQ(expected_vec, out);
}

TEST(leontev_n_mat_vec_seq, double_mat_vec) {
  // Create data
  std::vector<double> invec = {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0};
  std::vector<double> inmat(100, 10.0);
  const std::vector<double> expected_vec(10, 0.0);
  std::vector<double> out(10, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<double>(taskDataSeq, invec, inmat, out);

  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<double> matVecSequential(taskDataSeq);
  ASSERT_TRUE(matVecSequential.validation());
  matVecSequential.pre_processing();
  matVecSequential.run();
  matVecSequential.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i], expected_vec[i], 1e-6);
  }
}

TEST(leontev_n_mat_vec_seq, float_mat_vec) {
  // Create data
  std::vector<float> invec = {-10.0F, 10.0F, -10.0F, 10.0F, -10.0F, 10.0F, -10.0F, 10.0F, -10.0F, 10.0F};
  std::vector<float> inmat(100, 10.0F);
  const std::vector<float> expected_vec(10, 0.0F);
  std::vector<float> out(10, 0.0F);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<float>(taskDataSeq, invec, inmat, out);

  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<float> matVecSequential(taskDataSeq);
  ASSERT_TRUE(matVecSequential.validation());
  matVecSequential.pre_processing();
  matVecSequential.run();
  matVecSequential.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_NEAR(out[i], expected_vec[i], 1e-6);
  }
}

TEST(leontev_n_mat_vec_seq, int32_mat_vec) {
  // Create data
  std::vector<int32_t> invec(1000, 5);
  std::vector<int32_t> inmat(1000000, 5);
  std::vector<int32_t> out(1000, 0);
  const std::vector<int32_t> expected_vec(1000, 25000);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, invec, inmat, out);

  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<int32_t> matVecSequential(taskDataSeq);
  ASSERT_TRUE(matVecSequential.validation());
  matVecSequential.pre_processing();
  matVecSequential.run();
  matVecSequential.post_processing();
  ASSERT_EQ(out, expected_vec);
}

TEST(leontev_n_mat_vec_seq, uint32_mat_vec) {
  // Create data
  std::vector<uint32_t> invec(256, 2);
  std::vector<uint32_t> inmat(65536, 2);
  std::vector<uint32_t> out(256, 0);
  const std::vector<uint32_t> expected_vec(256, 1024);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<uint32_t>(taskDataSeq, invec, inmat, out);

  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<uint32_t> matVecSequential(taskDataSeq);
  ASSERT_TRUE(matVecSequential.validation());
  matVecSequential.pre_processing();
  matVecSequential.run();
  matVecSequential.post_processing();
  ASSERT_EQ(out, expected_vec);
}

TEST(leontev_n_mat_vec_seq, empty_array_mat_vec) {
  // Create data
  std::vector<int32_t> invec;
  std::vector<int32_t> inmat;
  const std::vector<int32_t> expected_vec;
  std::vector<int32_t> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, invec, inmat, out);

  // Create Task
  leontev_n_mat_vec_seq::MatVecSequential<int32_t> matVecSequential(taskDataSeq);
  ASSERT_FALSE(matVecSequential.validation());
}
