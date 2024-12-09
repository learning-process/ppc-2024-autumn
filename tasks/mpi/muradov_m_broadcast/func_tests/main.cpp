#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <random>

#include "boost/mpi/collectives/broadcast.hpp"
#include "mpi/muradov_m_broadcast/include/ops_mpi.hpp"

namespace muradov_m_broadcast_mpi {

std::vector<int> gen_rundom_vector(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> vector(n);
  for (int i = 0; i < n; i++) {
    vector[i] = dist(gen);
  }
  return vector;
}

void test_template_rundom(int n, int source = 0) {
  boost::mpi::communicator world;

  if (world.size() <= source) {
    GTEST_SKIP();
  }

  std::vector<int> A;
  std::vector<int> A_res(n);
  std::vector<int> B_res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(B_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  auto taskBroadcast = std::make_shared<BroadcastParallelMPI>(taskDataBroadcast);
  bool val_res = taskBroadcast->validation();
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  if (val_res) EXPECT_EQ(A_res, B_res);
}

}  // namespace muradov_m_broadcast_mpi

TEST(muradov_m_broadcast_mpi, data_size_0) { muradov_m_broadcast_mpi::test_template_rundom(0); }

TEST(muradov_m_broadcast_mpi, data_size_1) { muradov_m_broadcast_mpi::test_template_rundom(1); }

TEST(muradov_m_broadcast_mpi, data_size_2) { muradov_m_broadcast_mpi::test_template_rundom(2); }

TEST(muradov_m_broadcast_mpi, data_size_3) { muradov_m_broadcast_mpi::test_template_rundom(3); }

TEST(muradov_m_broadcast_mpi, data_size_4) { muradov_m_broadcast_mpi::test_template_rundom(4); }

TEST(muradov_m_broadcast_mpi, data_size_5) { muradov_m_broadcast_mpi::test_template_rundom(5); }

TEST(muradov_m_broadcast_mpi, data_size_6) { muradov_m_broadcast_mpi::test_template_rundom(6); }

TEST(muradov_m_broadcast_mpi, data_size_7) { muradov_m_broadcast_mpi::test_template_rundom(7); }

TEST(muradov_m_broadcast_mpi, data_size_8) { muradov_m_broadcast_mpi::test_template_rundom(8); }

TEST(muradov_m_broadcast_mpi, data_size_9) { muradov_m_broadcast_mpi::test_template_rundom(9); }

TEST(muradov_m_broadcast_mpi, data_size_10) { muradov_m_broadcast_mpi::test_template_rundom(10); }

TEST(muradov_m_broadcast_mpi, data_size_11) { muradov_m_broadcast_mpi::test_template_rundom(11); }

TEST(muradov_m_broadcast_mpi, data_size_13) { muradov_m_broadcast_mpi::test_template_rundom(13); }

TEST(muradov_m_broadcast_mpi, data_size_15) { muradov_m_broadcast_mpi::test_template_rundom(15); }

TEST(muradov_m_broadcast_mpi, data_size_20) { muradov_m_broadcast_mpi::test_template_rundom(20); }

TEST(muradov_m_broadcast_mpi, data_size_30) { muradov_m_broadcast_mpi::test_template_rundom(30); }

TEST(muradov_m_broadcast_mpi, data_size_40) { muradov_m_broadcast_mpi::test_template_rundom(40); }

TEST(muradov_m_broadcast_mpi, data_size_50) { muradov_m_broadcast_mpi::test_template_rundom(50); }

TEST(muradov_m_broadcast_mpi, source_1_data_size_50) { muradov_m_broadcast_mpi::test_template_rundom(50, 1); }

TEST(muradov_m_broadcast_mpi, source_2_data_size_10) { muradov_m_broadcast_mpi::test_template_rundom(50, 2); }

TEST(muradov_m_broadcast_mpi, rundom_source_between_0_and_world_size_minus_1) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  int n = 50;
  int source;
  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, world.size() - 1);
    std::vector<int> vector(n);
    source = dist(gen);
  }
  boost::mpi::broadcast(world, source, 0);

  std::vector<int> A;
  std::vector<int> A_res(n);
  std::vector<int> B_res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = muradov_m_broadcast_mpi::gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(B_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  auto taskBroadcast = std::make_shared<muradov_m_broadcast_mpi::BroadcastParallelMPI>(taskDataBroadcast);
  taskBroadcast->validation();
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  EXPECT_EQ(A_res, B_res);
}
