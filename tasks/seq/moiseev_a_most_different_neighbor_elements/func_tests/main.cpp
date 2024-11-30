#include <gtest/gtest.h>

#include "seq/moiseev_a_most_different_neighbor_elements/include/ops_seq.hpp"

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_int32_t) {
  std::vector<int32_t> in(1256, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = 2 * i;
  }
  in[234] = 0;
  in[235] = 4000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int32_t> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 4000);
  EXPECT_EQ(out_index[0], 234ull);
  EXPECT_EQ(out_index[1], 235ull);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_validate_func) {
  std::vector<int32_t> in(125, 1);
  std::vector<int32_t> out(2, 0);
  std::vector<uint64_t> out_index(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int32_t> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, false);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_double) {
  std::vector<double> in(25680, 1);
  std::vector<double> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i;
  }
  in[189] = -1000.1;
  in[190] = 9000.9;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<double> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_NEAR(out[0], -1000.1, 1e-6);
  EXPECT_NEAR(out[1], 9000.9, 1e-6);
  EXPECT_EQ(out_index[0], 189ull);
  EXPECT_EQ(out_index[1], 190ull);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_int8_t) {
  std::vector<int8_t> in(250, -1);
  std::vector<int8_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    if (i % 2 == 0) {
      in[i] = -50;
    } else {
      in[i] = 50;
    }
  }
  in[5] = 56;
  in[6] = -56;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int8_t> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_EQ(out[0], 56);
  EXPECT_EQ(out[1], -56);
  EXPECT_EQ(out_index[0], 5ull);
  EXPECT_EQ(out_index[1], 6ull);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_int64_t) {
  std::vector<int64_t> in(75836, 1);
  std::vector<int64_t> out(2, 0);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    if (i % 3 == 0) {
      in[i] = 10;
    }
    if (i % 3 == 1) {
      in[i] = 30;
    }
    if (i % 3 == 2) {
      in[i] = 70;
    }
  }
  in[20] = -1000;
  in[21] = 1119;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<int64_t> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_EQ(out[0], -1000);
  EXPECT_EQ(out[1], 1119);
  EXPECT_EQ(out_index[0], 20ull);
  EXPECT_EQ(out_index[1], 21ull);
}

TEST(moiseev_a_most_different_neighbor_elements_seq_test, check_float) {
  std::vector<float> in(20, 1.f);
  std::vector<float> out(2, 0.f);
  std::vector<uint64_t> out_index(2, 0);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] += (i + 1.f) * 2.5f;
  }
  in[0] = 110.001f;
  in[1] = -990.0025f;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_index.data()));
  taskData->outputs_count.emplace_back(out_index.size());

  moiseev_a_most_different_neighbor_elements_seq::MostDifferentNeighborElementsSequential<float> testTask(taskData);
  bool isValid = testTask.validation();
  EXPECT_EQ(isValid, true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  EXPECT_NEAR(out[0], 110.001f, 1e-6);
  EXPECT_NEAR(out[1], -990.0025f, 1e-6);
  EXPECT_EQ(out_index[0], 0ull);
  EXPECT_EQ(out_index[1], 1ull);
}
