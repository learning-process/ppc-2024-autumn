#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "boost/mpi/collectives/broadcast.hpp"
#include "mpi/burykin_m_broadcast_mpi/include/ops_mpi.hpp"

namespace burykin_m_broadcast_mpi_mpi {

void fillVector(std::vector<int>& vector, int min_val, int max_val) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min_val, max_val);
  for (size_t iter = 0; iter < vector.size();) {
    vector[iter++] = dist(gen);
  }
}

void test_template(int data_size, int source_worker = 0, int min_val = -100, int max_val = 100) {
  boost::mpi::communicator world;

  if (world.size() <= source_worker) GTEST_SKIP();

  std::vector<int> recv_vorker(data_size);
  std::vector<int> result_vector(data_size);
  int global_max;

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();

  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source_worker));
  task->inputs_count.emplace_back(1);

  if (world.rank() == source_worker) {
    burykin_m_broadcast_mpi_mpi::fillVector(recv_vorker, min_val, max_val);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(recv_vorker.data()));
    task->inputs_count.emplace_back(recv_vorker.size());
  }

  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task->outputs_count.emplace_back(result_vector.size());

  if (world.rank() == source_worker) {
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_max));
    task->outputs_count.emplace_back(1);
  }

  auto taskForProcessing = std::make_shared<StdBroadcastMPI>(task);
  bool val_res = taskForProcessing->validation();
  boost::mpi::broadcast(world, val_res, source_worker);
  if (val_res) {
    taskForProcessing->pre_processing();
    taskForProcessing->run();
    taskForProcessing->post_processing();

    boost::mpi::broadcast(world, recv_vorker, source_worker);

    EXPECT_EQ(recv_vorker, result_vector);
    if (world.size() == source_worker) {
      int result_max = *std::max_element(recv_vorker.begin(), recv_vorker.end());
      EXPECT_EQ(global_max, result_max);
    }
  }
}

}  // namespace burykin_m_broadcast_mpi_mpi

TEST(burykin_m_broadcast_mpi_mpi, DataSize1_Source0) { burykin_m_broadcast_mpi_mpi::test_template(1, 0, -10, 10); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize2_Source0) { burykin_m_broadcast_mpi_mpi::test_template(2, 0, 0, 100); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize3_Source0) { burykin_m_broadcast_mpi_mpi::test_template(3, 0, -100, 0); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize4_Source0) { burykin_m_broadcast_mpi_mpi::test_template(4, 0, -50, 50); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize10_Source1) { burykin_m_broadcast_mpi_mpi::test_template(10, 1, -100, 100); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize15_Source2) { burykin_m_broadcast_mpi_mpi::test_template(15, 2, -200, 200); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize20_Source0) { burykin_m_broadcast_mpi_mpi::test_template(20, 0, -500, 500); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize30_Source0) {
  burykin_m_broadcast_mpi_mpi::test_template(30, 0, -1000, 1000);
}

TEST(burykin_m_broadcast_mpi_mpi, DataSize40_Source1) { burykin_m_broadcast_mpi_mpi::test_template(40, 1, -100, 100); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize50_Source2) { burykin_m_broadcast_mpi_mpi::test_template(50, 2, -500, 0); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize0_Source0) { burykin_m_broadcast_mpi_mpi::test_template(0, 0, -10, 10); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize1_Source0_ZeroRange) {
  burykin_m_broadcast_mpi_mpi::test_template(1, 0, 0, 0);
}

TEST(burykin_m_broadcast_mpi_mpi, DataSize100_Source0) { burykin_m_broadcast_mpi_mpi::test_template(100, 0, -1, 1); }

TEST(burykin_m_broadcast_mpi_mpi, DataSize10_Rundom_Source) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP();
  }

  int source_worker;
  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, world.size() - 1);

    source_worker = dist(gen);
  }
  boost::mpi::broadcast(world, source_worker, 0);
  int data_size = 10;
  int min_val = -10;
  int max_val = 10;

  std::vector<int> recv_vorker(data_size);
  std::vector<int> result_vector(data_size);
  int global_max;

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();

  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source_worker));
  task->inputs_count.emplace_back(1);

  if (world.rank() == source_worker) {
    burykin_m_broadcast_mpi_mpi::fillVector(recv_vorker, min_val, max_val);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(recv_vorker.data()));
    task->inputs_count.emplace_back(recv_vorker.size());
  }

  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task->outputs_count.emplace_back(result_vector.size());

  if (world.rank() == source_worker) {
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_max));
    task->outputs_count.emplace_back(1);
  }

  auto taskForProcessing = std::make_shared<burykin_m_broadcast_mpi_mpi::StdBroadcastMPI>(task);
  bool val_res = taskForProcessing->validation();
  boost::mpi::broadcast(world, val_res, source_worker);
  if (val_res) {
    taskForProcessing->pre_processing();
    taskForProcessing->run();
    taskForProcessing->post_processing();

    boost::mpi::broadcast(world, recv_vorker, source_worker);

    EXPECT_EQ(recv_vorker, result_vector);
    if (world.size() == source_worker) {
      int result_max = *std::max_element(recv_vorker.begin(), recv_vorker.end());
      EXPECT_EQ(global_max, result_max);
    }
  }
}
