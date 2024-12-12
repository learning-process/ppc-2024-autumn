#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/zaitsev_a_scatter_handwritten/include/ops_mpi.hpp"

namespace zaitsev_a_scatter_handwritten {
std::vector<int> get_random_int_vector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) vec[i] = min + gen() % (max - min + 1);
  return vec;
}

std::vector<double> get_random_double_vector(int sz, double min, double max) {
  std::uniform_real_distribution<double> unif(min, max);
  std::default_random_engine re;
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; i++) vec[i] = unif(re);
  return vec;
}

template <auto func>
  requires std::same_as<decltype(+func),
                        int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm)>
void test_int(int sz, int min = -10e2, int max = -10e2) {
  std::vector<int> inp;
  std::vector<int> task_result(1, 0);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inp = zaitsev_a_scatter_handwritten::get_random_int_vector(sz, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(task_result.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  zaitsev_a_scatter::ScatterTask<int, func> task(taskDataPar, 0, MPI_INT);

  if (sz == 0) {
    if (world.rank() == 0) {
      ASSERT_FALSE(task.validation());
    }
    return;
  }

  EXPECT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(*std::min_element(inp.begin(), inp.end()), task_result[0]);
  }
}

template <auto func>
  requires std::same_as<decltype(+func),
                        int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm)>
void test_double(int sz, double min = -10e2, double max = 10e2) {
  std::vector<double> inp;
  std::vector<double> task_result(1, 0);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inp = zaitsev_a_scatter_handwritten::get_random_double_vector(sz, min, max);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(task_result.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  zaitsev_a_scatter::ScatterTask<double, func> task(taskDataPar, 0, MPI_DOUBLE);

  if (sz == 0) {
    if (world.rank() == 0) {
      ASSERT_FALSE(task.validation());
    }
    return;
  }

  EXPECT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    ASSERT_DOUBLE_EQ(*std::min_element(inp.begin(), inp.end()), task_result[0]);
  }
}
}  // namespace zaitsev_a_scatter_handwritten

TEST(zaitsev_a_scatter_handwritten, test__dtype_int__size_0) {
  zaitsev_a_scatter_handwritten::test_int<zaitsev_a_scatter::scatter>(0);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_int__size_1e0) {
  zaitsev_a_scatter_handwritten::test_int<zaitsev_a_scatter::scatter>(1e0);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_int__size_1e2) {
  zaitsev_a_scatter_handwritten::test_int<zaitsev_a_scatter::scatter>(1e2);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_int__size_1e5) {
  zaitsev_a_scatter_handwritten::test_int<zaitsev_a_scatter::scatter>(1e5);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_int__size_1e7) {
  zaitsev_a_scatter_handwritten::test_int<zaitsev_a_scatter::scatter>(1e7);
}

TEST(zaitsev_a_scatter_handwritten, test__dtype_double__size_0) {
  zaitsev_a_scatter_handwritten::test_double<zaitsev_a_scatter::scatter>(0);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_double__size_1e0) {
  zaitsev_a_scatter_handwritten::test_double<zaitsev_a_scatter::scatter>(1);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_double__size_1e2) {
  zaitsev_a_scatter_handwritten::test_double<zaitsev_a_scatter::scatter>(1e2);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_double__size_1e3) {
  zaitsev_a_scatter_handwritten::test_double<zaitsev_a_scatter::scatter>(1e3);
}
TEST(zaitsev_a_scatter_handwritten, test__dtype_double__size_1e5) {
  zaitsev_a_scatter_handwritten::test_double<zaitsev_a_scatter::scatter>(1e5);
}