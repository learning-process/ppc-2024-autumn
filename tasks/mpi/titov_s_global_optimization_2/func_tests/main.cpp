// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/titov_s_global_optimization_2/include/ops_mpi.hpp"

TEST(titov_s_global_optimization_2_mpi, Test_rectangle) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x + p.y * p.y; };
  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.x; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.y; };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}

TEST(titov_s_global_optimization_2_mpi, Test_triangle) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) {
        return 10.0 * (p.x - 3.5) * (p.x - 3.5) + 20.0 * (p.y - 4.0) * (p.y - 4.0);
      };
  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 6.0 - (p.x + p.y); };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return (2.0 * p.x + p.y) - 6; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 1.0 - (p.x - p.y); };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return (0.5 * p.x - p.y) + 4; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}

TEST(titov_s_global_optimization_2_mpi, Test_triangle_2) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x - 12.0 * p.x + p.y * p.y - 4.0 * p.y; };
  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 2.0 * p.x + 4.0 - p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.x - p.y; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y - (0.2 * p.x + 0.4); };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{constraint1, constraint2,
                                                                                              constraint3});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}

TEST(titov_s_global_optimization_2_mpi, Test_4) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x - 20.0 * p.x + p.y * p.y - 6.0 * p.y; };

  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 3.0 + p.x + 2.0 * p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 4.0 - p.x + p.y; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 3.0 - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{constraint1, constraint2,
                                                                                              constraint3});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}

TEST(titov_s_global_optimization_2_mpi, Test_5_constraits_non_linear) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) {
        return (p.x - 5.0) * (p.x - 5.0) + (p.y - 5.0) * (p.y - 5.0);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 5.0 - p.x - p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y * p.y * p.y - p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x; };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y; };
  auto constraint5 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x * p.x * p.x - p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4, constraint5});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}

TEST(titov_s_global_optimization_2_mpi, Test_constraits_cubic) {
  boost::mpi::communicator world;
  double step_size = 0.05;
  double tolerance = 0.0001;
  size_t max_iterations = 100;
  std::function<double(const titov_s_global_optimization_2_mpi::Point&)> func =
      [](const titov_s_global_optimization_2_mpi::Point& p) {
        return (p.x - 3.0) * (p.x - 3.0) + (p.y - 3.0) * (p.y - 3.0);
      };

  auto constraint1 = [](const titov_s_global_optimization_2_mpi::Point& p) { return 5.0 - p.x - p.y; };
  auto constraint2 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y * p.y * p.y - p.x; };
  auto constraint3 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.x; };
  auto constraint4 = [](const titov_s_global_optimization_2_mpi::Point& p) { return p.y; };

  auto constraints_ptr =
      std::make_shared<std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>>(
          std::vector<std::function<double(const titov_s_global_optimization_2_mpi::Point&)>>{
              constraint1, constraint2, constraint3, constraint4});

  std::vector<titov_s_global_optimization_2_mpi::Point> outPar(1, {0.0, 0.0});
  std::vector<titov_s_global_optimization_2_mpi::Point> outSeq;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
  taskDataPar->inputs_count.emplace_back(constraints_ptr->size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step_size));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_iterations));
  taskDataPar->inputs_count.emplace_back(3);
  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outPar.data()));
    taskDataPar->outputs_count.emplace_back(outPar.size());
  }

  titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel optimizationTaskPar(taskDataPar);
  ASSERT_TRUE(optimizationTaskPar.validation());
  optimizationTaskPar.pre_processing();
  optimizationTaskPar.run();
  optimizationTaskPar.post_processing();

  if (world.rank() == 0) {
    outSeq = std::vector<titov_s_global_optimization_2_mpi::Point>(1, {0, 0});

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraints_ptr.get()));
    taskDataSeq->inputs_count.emplace_back(constraints_ptr->size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outSeq.data()));
    taskDataSeq->outputs_count.emplace_back(outSeq.size());

    titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential optimizationTaskSeq(taskDataSeq);

    ASSERT_TRUE(optimizationTaskSeq.validation());
    optimizationTaskSeq.pre_processing();
    optimizationTaskSeq.run();
    optimizationTaskSeq.post_processing();

    ASSERT_NEAR(outPar[0].x, outSeq[0].x, 0.2);
    ASSERT_NEAR(outPar[0].y, outSeq[0].y, 0.2);
  }
}
