// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/titov_s_global_optimization_2/include/ops_seq.hpp"

TEST(titov_s_global_optimization_2_seq, Test_0) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x + p.y * p.y; };
  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.x; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 0, 0.1);
  ASSERT_NEAR(out[0].y, 0, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Test_1) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) {
        return 10.0 * (p.x - 3.5) * (p.x - 3.5) + 20.0 * (p.y - 4.0) * (p.y - 4.0);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 6.0 - (p.x + p.y); };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return (2.0 * p.x + p.y) - 6; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return 1.0 - (p.x - p.y); };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return (0.5 * p.x - p.y) + 4; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 2.5, 0.1);
  ASSERT_NEAR(out[0].y, 3.5, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Test_2) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x - 12.0 * p.x + p.y * p.y - 4.0 * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 2.0 * p.x + 4.0 - p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.x - p.y; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y - (0.2 * p.x + 0.4); };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{constraint1, constraint2,
                                                                                              constraint3});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 3.0, 0.1);
  ASSERT_NEAR(out[0].y, 1.0, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Test_3) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x - 20.0 * p.x + p.y * p.y - 6.0 * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.0 + p.x + 2.0 * p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return +4.0 - p.x + p.y; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{constraint1, constraint2,
                                                                                              constraint3});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 7.0, 0.1);
  ASSERT_NEAR(out[0].y, 3.0, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Test_4) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return std::sin(p.x) + std::cos(p.y); };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 2.0 - p.x; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.0 - p.x + p.y; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x; };
  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 0.0, 0.1);
  ASSERT_NEAR(out[0].y, 0.0, 0.15);
}

TEST(titov_s_global_optimization_2_seq, Test_6) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) {
        return std::log(1 + p.x * p.x) + std::log(1 + p.y * p.y);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return 10.0 - (p.x * p.x + p.y * p.y); };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x - 1.0; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{constraint1, constraint2,
                                                                                              constraint3});
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 1.0, 0.1);
  ASSERT_NEAR(out[0].y, 0.0, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Test_7) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x * p.x + p.y * p.y * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x - 2.9; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.1 - p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  optimizationTask.pre_processing();
  optimizationTask.run();
  optimizationTask.post_processing();

  ASSERT_NEAR(out[0].x, 2.9, 0.1);
  ASSERT_NEAR(out[0].y, 0.0, 0.1);
}

TEST(titov_s_global_optimization_2_seq, Validation_InvalidInputs) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x * p.x + p.y * p.y * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x - 2.9; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.1 - p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  EXPECT_FALSE(optimizationTask.validation());
}

TEST(titov_s_global_optimization_2_seq, Test_Validation_InvalidOutputs) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x * p.x + p.y * p.y * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x - 2.9; };
  auto constraint2 = [](const titov_s_global_optimization_2_seq::Point& p) { return 3.1 - p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return 4.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(1);

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  EXPECT_FALSE(optimizationTask.validation());
}

TEST(titov_s_global_optimization_2_seq, Test_Wrong_Constraits) {
  std::function<double(const titov_s_global_optimization_2_seq::Point&)> func =
      [](const titov_s_global_optimization_2_seq::Point& p) { return p.x * p.x * p.x + p.y * p.y * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_seq::Point& p) { return p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_seq::Point& p) { return -4.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_seq::Point&)>>{constraint1, constraint3,
                                                                                              constraint4});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

  std::vector<titov_s_global_optimization_2_seq::Point> out(1, {0.0, 0.0});
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  titov_s_global_optimization_2_seq::GlobalOpt2Sequential optimizationTask(taskDataSeq);

  ASSERT_TRUE(optimizationTask.validation());
  EXPECT_FALSE(optimizationTask.pre_processing());
}
