#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/zaitsev_a_scatter/include/ops_mpi.hpp"

namespace zaitsev_a_scatter {
std::vector<int> get_random_int_vector(int sz, int min, int max, int extrema = -100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) vec[i] = min + gen() % (max - min + 1);
  vec[0] = extrema;
  return vec;
}

std::vector<double> get_random_double_vector(int sz, double min, double max, double extrema) {
  std::uniform_real_distribution<double> unif(min, max);
  std::default_random_engine re;
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; i++) vec[i] = unif(re);
  vec[0] = extrema;
  return vec;
}
}  // namespace zaitsev_a_scatter

template <auto func>
  requires std::same_as<decltype(+func),
                        int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm)>
void test_int(int sz, int min = -1000, int max = 1000, int extrema = -5000) {
  std::vector<int> inp;
  std::vector<int> ref(1, 15);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inp = zaitsev_a_scatter::get_random_int_vector(sz, min, max, extrema);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  zaitsev_a_scatter::TestMPITaskParallel<int, func> testMpiTaskParallel(taskDataPar, 0, MPI_INT);
  if (!testMpiTaskParallel.validation()) {
    GTEST_SKIP();
    return;
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}

template <auto func>
  requires std::same_as<decltype(+func),
                        int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm)>
void test_double(int sz, double min = -10e3, double max = 10e3, double extrema = -10e4) {
  std::vector<double> inp;
  std::vector<double> ref(1, 0);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    inp = zaitsev_a_scatter::get_random_double_vector(sz, min, max, extrema);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp.data()));
    taskDataPar->inputs_count.emplace_back(sz);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  zaitsev_a_scatter::TestMPITaskParallel<double, func> testMpiTaskParallel(taskDataPar, 0, MPI_DOUBLE);
  if (!testMpiTaskParallel.validation()) {
    GTEST_SKIP();
    return;
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_DOUBLE_EQ(extrema, reinterpret_cast<double*>(taskDataPar->outputs[0])[0]);
  }
}

TEST(zaitsev_a_scatter, test__func_handwritten__dtype_int__size_1) { test_int<zaitsev_a_scatter::scatter>(1); }
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_int__size_1e2) { test_int<zaitsev_a_scatter::scatter>(1e2); }
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_int__size_1e5) { test_int<zaitsev_a_scatter::scatter>(1e5); }
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_int__size_1e7) { test_int<zaitsev_a_scatter::scatter>(1e7); }

TEST(zaitsev_a_scatter, test__func_handwritten__dtype_double__size_1) { test_double<zaitsev_a_scatter::scatter>(1); }
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_double__size_1e2) {
  test_double<zaitsev_a_scatter::scatter>(1e2);
}
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_double__size_1e3) {
  test_double<zaitsev_a_scatter::scatter>(1e3);
}
TEST(zaitsev_a_scatter, test__func_handwritten__dtype_double__size_1e5) {
  test_double<zaitsev_a_scatter::scatter>(1e5);
}

TEST(zaitsev_a_scatter, test__func_builtin__dtype_int__size_1) { test_int<MPI_Scatter>(1); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_int__size_1e2) { test_int<MPI_Scatter>(1e2); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_int__size_1e5) { test_int<MPI_Scatter>(1e5); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_int__size_1e7) { test_int<MPI_Scatter>(1e7); }

TEST(zaitsev_a_scatter, test__func_builtin__dtype_double__size_1) { test_double<MPI_Scatter>(1); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_double__size_1e2) { test_double<MPI_Scatter>(1e2); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_double__size_1e3) { test_double<MPI_Scatter>(1e3); }
TEST(zaitsev_a_scatter, test__func_builtin__dtype_double__size_1e5) { test_double<MPI_Scatter>(1e5); }
