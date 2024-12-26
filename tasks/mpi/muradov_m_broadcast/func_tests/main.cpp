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
  int my_global_sum_A;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&my_global_sum_A));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<MyBroadcastParallelMPI>(taskDataBroadcast);
  bool val_res = taskBroadcast->validation();
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  if (val_res) {
    boost::mpi::broadcast(world, A, source);
    EXPECT_EQ(A, A_res);

    if (world.rank() == source) {
      int check_result = std::accumulate(A.begin(), A.end(), 0);
      EXPECT_EQ(my_global_sum_A, check_result);
    }
  }
}

void test_template_rundom_mpi(int n, int source = 0) {
  boost::mpi::communicator world;

  if (world.size() <= source) {
    GTEST_SKIP();
  }

  std::vector<int> A;
  std::vector<int> A_res(n);
  int my_global_sum_A;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&my_global_sum_A));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<MpiBroadcastParallelMPI>(taskDataBroadcast);
  bool val_res = taskBroadcast->validation();
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  if (val_res) {
    boost::mpi::broadcast(world, A, source);
    EXPECT_EQ(A, A_res);

    if (world.rank() == source) {
      int check_result = std::accumulate(A.begin(), A.end(), 0);
      EXPECT_EQ(my_global_sum_A, check_result);
    }
  }
}

}  // namespace muradov_m_broadcast_mpi

// TEST(muradov_m_broadcast_mpi, data_size_0) { muradov_m_broadcast_mpi::test_template_rundom(0); }

TEST(muradov_m_broadcast_my, data_size_1) { muradov_m_broadcast_mpi::test_template_rundom(1); }
TEST(muradov_m_broadcast_mpi, data_size_1_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(1); }

TEST(muradov_m_broadcast_my, data_size_2) { muradov_m_broadcast_mpi::test_template_rundom(2); }
TEST(muradov_m_broadcast_mpi, data_size_2_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(2); }

TEST(muradov_m_broadcast_my, data_size_3) { muradov_m_broadcast_mpi::test_template_rundom(3); }
TEST(muradov_m_broadcast_mpi, data_size_3_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(3); }

TEST(muradov_m_broadcast_my, data_size_4) { muradov_m_broadcast_mpi::test_template_rundom(4); }
TEST(muradov_m_broadcast_mpi, data_size_4_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(4); }

TEST(muradov_m_broadcast_my, data_size_5) { muradov_m_broadcast_mpi::test_template_rundom(5); }
TEST(muradov_m_broadcast_mpi, data_size_5_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(5); }

TEST(muradov_m_broadcast_my, data_size_6) { muradov_m_broadcast_mpi::test_template_rundom(6); }
TEST(muradov_m_broadcast_mpi, data_size_6_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(6); }

TEST(muradov_m_broadcast_my, data_size_7) { muradov_m_broadcast_mpi::test_template_rundom(7); }
TEST(muradov_m_broadcast_mpi, data_size_7_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(7); }

TEST(muradov_m_broadcast_my, data_size_8) { muradov_m_broadcast_mpi::test_template_rundom(8); }
TEST(muradov_m_broadcast_mpi, data_size_8_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(8); }

TEST(muradov_m_broadcast_my, data_size_9) { muradov_m_broadcast_mpi::test_template_rundom(9); }
TEST(muradov_m_broadcast_mpi, data_size_9_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(9); }

TEST(muradov_m_broadcast_my, data_size_10) { muradov_m_broadcast_mpi::test_template_rundom(10); }
TEST(muradov_m_broadcast_mpi, data_size_10_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(10); }

TEST(muradov_m_broadcast_my, data_size_11) { muradov_m_broadcast_mpi::test_template_rundom(11); }
TEST(muradov_m_broadcast_mpi, data_size_11_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(11); }

TEST(muradov_m_broadcast_my, data_size_13) { muradov_m_broadcast_mpi::test_template_rundom(13); }
TEST(muradov_m_broadcast_mpi, data_size_13_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(13); }

TEST(muradov_m_broadcast_my, data_size_15) { muradov_m_broadcast_mpi::test_template_rundom(15); }
TEST(muradov_m_broadcast_mpi, data_size_15_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(15); }

TEST(muradov_m_broadcast_my, data_size_20) { muradov_m_broadcast_mpi::test_template_rundom(20); }
TEST(muradov_m_broadcast_mpi, data_size_20_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(20); }

TEST(muradov_m_broadcast_my, data_size_30) { muradov_m_broadcast_mpi::test_template_rundom(30); }
TEST(muradov_m_broadcast_mpi, data_size_30_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(30); }

TEST(muradov_m_broadcast_my, data_size_40) { muradov_m_broadcast_mpi::test_template_rundom(40); }
TEST(muradov_m_broadcast_mpi, data_size_40_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(40); }

TEST(muradov_m_broadcast_my, data_size_50) { muradov_m_broadcast_mpi::test_template_rundom(50); }
TEST(muradov_m_broadcast_mpi, data_size_50_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(50); }

TEST(muradov_m_broadcast_my, source_1_data_size_50) { muradov_m_broadcast_mpi::test_template_rundom(50, 1); }
TEST(muradov_m_broadcast_mpi, source_1_data_size_50_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(50, 1); }

TEST(muradov_m_broadcast_my, source_2_data_size_10) { muradov_m_broadcast_mpi::test_template_rundom(50, 2); }
TEST(muradov_m_broadcast_mpi, source_2_data_size_10_mpi) { muradov_m_broadcast_mpi::test_template_rundom_mpi(50, 2); }
