#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <random>

#include "boost/mpi/collectives/broadcast.hpp"
#include "mpi/durynichev_d_custom_allreduce/include/ops_mpi.hpp"

namespace durynichev_d_custom_allreduce_mpi {

std::vector<int> genRundomVector(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);
  std::vector<int> vector(n);
  for (int i = 0; i < n; i++) {
    vector[i] = dist(gen);
  }
  return vector;
}

void run_and_validation_test_template(int n) {
  boost::mpi::communicator world;

  std::vector<int> data;
  int global_sum;
  int check;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = genRundomVector(n);
    check = std::accumulate(data.begin(), data.end(), 0);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_sum));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<MyAllreduceMPI>(taskDataBroadcast);
  bool val = taskBroadcast->validation();
  boost::mpi::broadcast(world, val, 0);
  if (val) {
    taskBroadcast->pre_processing();
    taskBroadcast->run();
    taskBroadcast->post_processing();

    boost::mpi::broadcast(world, check, 0);
    EXPECT_EQ(global_sum, check);
  }
}

}  // namespace durynichev_d_custom_allreduce_mpi

TEST(durynichev_d_custom_allreduce_mpi, val_test_with_size_0) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(0);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_1) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(1);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_2) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(2);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_3) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(3);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_4) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(4);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_5) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(5);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_6) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(6);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_7) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(7);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_10) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(10);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_100) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(100);
}

TEST(durynichev_d_custom_allreduce_mpi, test_with_size_1024) {
  durynichev_d_custom_allreduce_mpi::run_and_validation_test_template(1024);
}