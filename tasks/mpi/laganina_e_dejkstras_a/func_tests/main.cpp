#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>
#include "mpi/laganina_e_dejkstras_a/include/ops_mpi.hpp"

namespace laganina_e_dejskras_a_mpi {

std::vector<int> getRandomgraph(int v) {
  std::vector<int> graph(v * v);
  std::mt19937 engine;
  for (std::size_t i = 0; i < v; i++) {
    for (size_t j = 0; j <= i; j++) {
      if (i == j) {
        graph[i * v + j] = 0;
        continue;
      }
      int tmp1 = engine() % 10;
      if (tmp1 < 0) {
        tmp1 *= -1;
      }
      graph[j * v + i] = graph[i * v + j] = tmp1;
    }
  }
  return graph;
}

}  // namespace laganina_e_dejskras_a_mpi

TEST(laganina_e_dejkstras_a_mpi, Test_validation) {
  boost::mpi::communicator world;
  int v_ = 0;
  std::vector<int> graph = {0, 1, 2, 0, 0, 1, 0, 0, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 2};
  std::vector<int> trueResult = {0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(v_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
    taskDataPar->outputs_count.emplace_back(trueResult.size());
  }
  // Create Task
  laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testTaskParallel.validation(), false);
  }
}

 TEST(laganina_e_dejkstras_a_mpi, Test_3_1) {
  boost::mpi::communicator world;
  int v_ = 3;
  std::vector<int> graph = {0, 1, 2, 0, 0, 1, 0, 0, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 2};
  std::vector<int> trueResult = {0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(v_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
    taskDataPar->outputs_count.emplace_back(trueResult.size());
  }
  // Create Task
  laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(expectResult, trueResult);
  }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_5_1) {
   boost::mpi::communicator world;
   int v_ = 5;
   std::vector<int> graph = {0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

   // Create data
   std::vector<int> expectResult = {0, 1, 2, 1, 2};
   std::vector<int> trueResult = {0, 0, 0, 0, 0};

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_4_2) {
   boost::mpi::communicator world;
   int v_ = 4;
   std::vector<int> graph = {0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   // Create data
   std::vector<int> expectResult = {0, 1, 1, 1};
   std::vector<int> trueResult = {0, 0, 0, 0};

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_3_random) {
   boost::mpi::communicator world;
   int v_ = 3;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult = {0, 0, 0};
   std::vector<int> trueResult = {0, 0, 0};

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_10_random) {
   boost::mpi::communicator world;
   int v_ = 10;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_13_random) {
   boost::mpi::communicator world;
   int v_ = 13;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_25_random) {
   boost::mpi::communicator world;
   int v_ = 25;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_55_random) {
   boost::mpi::communicator world;
   int v_ = 55;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_76_random) {
   boost::mpi::communicator world;
   int v_ = 76;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }

 TEST(laganina_e_dejkstras_a_mpi, Test_101_random) {
   boost::mpi::communicator world;
   int v_ = 101;
   std::vector<int> graph = laganina_e_dejskras_a_mpi::getRandomgraph(v_);
   for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
     graph[k] = 0;
   }

   // Create data
   std::vector<int> expectResult(v_, 0);
   std::vector<int> trueResult(v_, 0);

   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
   if (world.rank() == 0) {
     taskDataPar->inputs_count.emplace_back(v_);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
     taskDataPar->outputs_count.emplace_back(trueResult.size());
   }
   // Create Task
   laganina_e_dejskras_a_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
   ASSERT_EQ(testTaskParallel.validation(), true);
   testTaskParallel.pre_processing();
   testTaskParallel.run();
   testTaskParallel.post_processing();
   if (world.rank() == 0) {
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs_count.emplace_back(v_);
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expectResult.data()));
     taskDataSeq->outputs_count.emplace_back(expectResult.size());
     laganina_e_dejskras_a_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
     ASSERT_EQ(testTaskSequential.validation(), true);
     testTaskSequential.pre_processing();
     testTaskSequential.run();
     testTaskSequential.post_processing();
     ASSERT_EQ(expectResult, trueResult);
   }
 }