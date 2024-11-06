#include <gtest/gtest.h>

#include "mpi/baranov_a_ring_topology/include/header_topology.hpp"
template <typename tp>
typename std::enable_if<std::is_arithmetic<tp>::value>::type get_rnd_vec(std::vector<tp> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());

  if constexpr (std::is_integral<tp>::value) {
    // Для целых чисел
    std::uniform_int_distribution<tp> dist(0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  } else if constexpr (std::is_floating_point<tp>::value) {
    // Для вещественных чисел
    std::uniform_real_distribution<tp> dist(0.0, vec.size());
    std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
  }
}

TEST(baranov_a_ring_topology, Test_ring_0_int) {
  const int N = 0;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_100_int) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_1000_int) {
  const int N = 1000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_10000_int) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<int> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_100_double) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_1000_double) {
  const int N = 1000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_10000_double) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<double> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_100_uint) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<uint32_t> arr(N);
  std::vector<uint32_t> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<uint32_t> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}

TEST(baranov_a_ring_topology, Test_ring_10000_uint) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<uint32_t> arr(N);
  std::vector<uint32_t> out(N);
  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    get_rnd_vec(arr);
    data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data->inputs_count.emplace_back(arr.size());
    data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data->outputs_count.emplace_back(1);
  }
  baranov_a_ring_topology_mpi::ring_topology<uint32_t> test1(data);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  if (world.rank() == 0) ASSERT_EQ(true, std::equal(arr.begin(), arr.end(), out.begin(), out.end()));
}