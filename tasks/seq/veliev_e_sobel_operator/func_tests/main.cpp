// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <vector>

#include "seq/veliev_e_sobel_operator/include/ops_seq.hpp"

TEST(veliev_e_sobel_operator_seq, Teststart) {
  // Create data
  int h = 5;
  int w = 5;
  std::vector<double> in = {220, 220, 220, 50,  50,  220, 220, 220, 50,  50,  220, 220, 220,
                            50,  50,  220, 220, 220, 50,  50,  220, 220, 220, 50,  50};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  veliev_e_sobel_operator_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  /* std::cout << std::fixed << std::setprecision(3);
   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << in[i * w + j];
     }
     std::cout << std::endl;
   }

   std::cout << std::endl;

   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << out[i * w + j];
     }
     std::cout << std::endl;
   }*/

  for (size_t i = 0; i < out.size(); i++) ASSERT_NEAR(out[i], outex[i], 1e-10);
}

TEST(veliev_e_sobel_operator_seq, Teststart1) {
  // Create data
  int h = 7;
  int w = 7;
  std::vector<double> in = {220, 220, 220, 220, 50,  50, 50, 220, 220, 220, 50, 50, 50, 50, 220, 220, 50,
                            50,  50,  50,  50,  220, 50, 50, 50,  50,  50,  50, 50, 50, 50, 50,  50,  50,
                            50,  50,  50,  50,  50,  50, 50, 50,  50,  50,  50, 50, 50, 50, 50};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.33333, 1.00000,
                               1.00000, 0.33333, 0.00000, 0.00000, 0.00000, 1.00000, 1.00000, 0.33333, 0.00000, 0.00000,
                               0.00000, 0.00000, 1.00000, 0.33333, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.33333,
                               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
                               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  veliev_e_sobel_operator_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  /*std::cout << std::fixed << std::setprecision(3);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      std::cout << std::setw(8) << in[i * w + j];
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      std::cout << std::setw(8) << out[i * w + j];
    }
    std::cout << std::endl;
  }*/

  for (size_t i = 0; i < out.size(); i++) ASSERT_NEAR(out[i], outex[i], 0.00001);
}

TEST(veliev_e_sobel_operator_seq, Teststart2) {
  // Create data
  int h = 3;
  int w = 3;
  std::vector<double> in = {100, 200, 100, 150, 250, 150, 100, 200, 100};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  veliev_e_sobel_operator_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  /*std::cout << std::fixed << std::setprecision(3);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      std::cout << std::setw(8) << in[i * w + j];
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      std::cout << std::setw(8) << out[i * w + j];
    }
    std::cout << std::endl;
  }*/

  for (size_t i = 0; i < out.size(); i++) ASSERT_NEAR(out[i], outex[i], 0.00001);
}

TEST(veliev_e_sobel_operator_seq, Teststart3) {
  // Create data
  int h = 4;
  int w = 5;
  std::vector<double> in = {10, 50, 100, 50, 10, 20, 60, 120, 60, 20, 10, 50, 100, 50, 10, 0, 40, 80, 40, 0};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 1.000, 0.000,
                               0.000, 0.983, 0.316, 0.983, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  veliev_e_sobel_operator_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  /* std::cout << std::fixed << std::setprecision(3);
   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << in[i * w + j];
     }
     std::cout << std::endl;
   }

   std::cout << std::endl;

   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << out[i * w + j];
     }
     std::cout << std::endl;
   }*/

  for (size_t i = 0; i < out.size(); i++) ASSERT_NEAR(out[i], outex[i], 0.001);
}

TEST(veliev_e_sobel_operator_seq, Teststart4) {
  // Create data
  int h = 9;
  int w = 10;
  std::vector<double> in = {255, 255, 255, 128, 64,  32,  32,  64,  128, 255, 255, 255, 200, 128, 64, 32, 32,  64,
                            200, 255, 255, 200, 128, 100, 50,  25,  25,  50,  100, 200, 128, 100, 64, 50, 25,  12,
                            12,  25,  50,  64,  64,  50,  32,  25,  12,  6,   6,   12,  25,  32,  64, 50, 32,  25,
                            12,  6,   6,   12,  25,  32,  128, 100, 64,  50,  25,  12,  12,  25,  50, 64, 255, 200,
                            128, 100, 50,  25,  25,  50,  100, 200, 255, 255, 200, 128, 64,  32,  32, 64, 200, 255};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {
      0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.447, 0.784, 0.768, 0.492,
      0.168, 0.168, 0.682, 0.979, 0.000, 0.000, 0.912, 0.841, 0.625, 0.446, 0.183, 0.183, 0.579, 1.000, 0.000,
      0.000, 0.872, 0.632, 0.446, 0.321, 0.148, 0.148, 0.321, 0.579, 0.000, 0.000, 0.338, 0.249, 0.183, 0.148,
      0.058, 0.058, 0.148, 0.183, 0.000, 0.000, 0.338, 0.249, 0.183, 0.148, 0.058, 0.058, 0.148, 0.183, 0.000,
      0.000, 0.872, 0.632, 0.446, 0.321, 0.148, 0.148, 0.321, 0.579, 0.000, 0.000, 0.912, 0.841, 0.625, 0.446,
      0.183, 0.183, 0.579, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  veliev_e_sobel_operator_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  /* std::cout << std::fixed << std::setprecision(3);
   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << in[i * w + j];
     }
     std::cout << std::endl;
   }

   std::cout << std::endl;

   for (int i = 0; i < h; ++i) {
     for (int j = 0; j < w; ++j) {
       std::cout << std::setw(8) << out[i * w + j];
     }
     std::cout << std::endl;
   }*/

  for (size_t i = 0; i < out.size(); i++) ASSERT_NEAR(out[i], outex[i], 0.001);
}
