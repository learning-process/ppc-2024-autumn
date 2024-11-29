// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/vladimirova_j_gather/include/ops_seq.hpp"

TEST(Sequential_Operations, vladimirova_j_forward_backward_test) {
  std::vector<int> global_vector = {-2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_seq::TestTaskSequential testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}

TEST(Sequential_Operations, vladimirova_j_right_left_test) {
  std::vector<int> global_vector = {-1, 1, -1, 1, -1, 1, -1, 1, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_seq::TestTaskSequential testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations, vladimirova_j_more_dead_ends_test) {
  std::vector<int> global_vector = {1,  2, 2, 1, 2,  1,  -1, -1, -1, 2, 1, -2, 2, 2,  1,  2,  1, 2,
                                    -2, 1, 2, 1, -2, -1, -1, 2,  1,  2, 1, 1,  2, -1, -1, -2, 1, 2};
  // 1 2 2    1 -1 -1 1    -2   2 2 1 2 1 2 -2 1 2 1 -2 -1 -1 2 1 1 -1 -2 1 2
  std::vector<int32_t> ans_vec = {1, 2, 2, 2, 1, 2, 1, 1, 2, 1, -2, -1, -1, 2, 1, -2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_seq::TestTaskSequential testTaskSequential(taskDataPar);
  testTaskSequential.validation();
  // ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations, vladimirova_j_random_test) {
  std::vector<int> some_dead_end;
  std::vector<int> tmp;
  std::vector<int> global_vector;
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};

  int noDEnd = 0;
  for (int j = 0; j < 10; j++) {
    some_dead_end = vladimirova_j_gather_seq::getRandomVector(5);
    tmp = vladimirova_j_gather_seq::getRandomVector(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i : some_dead_end) {
      if ((i != 2) && (i != -2)) i *= -1;
    }
    for (int i = some_dead_end.size() - 1; i >= 0; i--) global_vector.push_back(some_dead_end[i]);
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_seq::TestTaskSequential testTaskSequential(taskDataPar);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ((int)taskDataPar->outputs_count[0] <= noDEnd, true);
}

TEST(Sequential_Operations, vladimirova_j_1_test) {
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 2, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_seq::TestTaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ((int)taskDataSeq->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
