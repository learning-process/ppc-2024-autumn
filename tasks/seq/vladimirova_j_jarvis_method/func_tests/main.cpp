// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/vladimirova_j_jarvis_method/include/ops_seq.hpp"
#include "seq/vladimirova_j_jarvis_method/func_tests/test_val.cpp"

TEST(Sequential, Test_10_0) {
  const int n = 10;
  // Create data
  std::vector<int> in = data_10_0;
  std::vector<int> ans = ans_data_10_0;
  std::vector<int> out(ans.size());

  for (auto i : data_10_0) std::cout << i << ". ";
  std::cout << std::endl;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < taskDataSeq->outputs_count[0]; i++) std::cout << out[i] << ". ";
  std::cout << std::endl;

  ASSERT_EQ(ans, out);

}


TEST(Sequential, Test_10_1) {
	const int n = 10;
	// Create data
	std::vector<int> in = data_10_1;
	std::vector<int> out(ans_data_10_1.size());

	for (auto i : data_10_0) std::cout << i << ". ";
	std::cout << std::endl;
	// Create TaskData
	std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	taskDataSeq->inputs_count.emplace_back(n);
	taskDataSeq->inputs_count.emplace_back(n);
	taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	taskDataSeq->outputs_count.emplace_back(n);

	// Create Task
	vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
	ASSERT_EQ(testTaskSequential.validation(), true);
	testTaskSequential.pre_processing();
	testTaskSequential.run();
	testTaskSequential.post_processing();

	for (size_t i = 0; i < taskDataSeq->outputs_count[0]; i++) std::cout << out[i] << ". ";
	std::cout << std::endl;

	ASSERT_EQ(ans_data_10_1[0], out[0]);
}




TEST(Sequential, Test_10_2) {
	const int n = 10;
	// Create data
	std::vector<int> in = data_10_2;
	std::vector<int> out(ans_data_10_2.size());

	for (auto i : data_10_0) std::cout << i << ". ";
	std::cout << std::endl;
	// Create TaskData
	std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
	taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
	taskDataSeq->inputs_count.emplace_back(n);
	taskDataSeq->inputs_count.emplace_back(n);
	taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
	taskDataSeq->outputs_count.emplace_back(n);

	// Create Task
	vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
	ASSERT_EQ(testTaskSequential.validation(), true);
	testTaskSequential.pre_processing();
	testTaskSequential.run();
	testTaskSequential.post_processing();

	for (size_t i = 0; i < taskDataSeq->outputs_count[0]; i++) std::cout << out[i] << ". ";
	std::cout << std::endl;

	ASSERT_EQ(ans_data_10_2[0], out[0]);
}

