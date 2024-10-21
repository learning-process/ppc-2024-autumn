#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"

#define EXPAND(x) x

#define T_DEF(macro, ...)             \
  EXPAND(macro(int16_t, __VA_ARGS__)) \
  EXPAND(macro(int32_t, __VA_ARGS__)) \
  EXPAND(macro(int64_t, __VA_ARGS__)) \
  EXPAND(macro(float, __VA_ARGS__))

using CountType = uint32_t;

class krylov_m_num_of_alternations_signs_mpi_test : public ::testing::Test {
 protected:
  template <typename ElementType>
  void run_generic_test(const boost::mpi::communicator &world, const CountType count, std::vector<ElementType> &in,
                        const std::vector<CountType> &shift_indices, CountType &out,
                        std::shared_ptr<ppc::core::TaskData> &taskDataPar) {
    if (world.rank() == 0) {
      in = std::vector<ElementType>(count);
      std::iota(in.begin(), in.end(), 1);

      for (auto idx : shift_indices) {
        in[idx] *= -1;
      }

      //
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataPar->inputs_count.emplace_back(in.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskDataPar->outputs_count.emplace_back(1);
    }

    //
    krylov_m_num_of_alternations_signs_mpi::TestMPITaskParallel<ElementType, CountType> testMpiTaskParallel(
        taskDataPar);
    ASSERT_TRUE(testMpiTaskParallel.validation());
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
  }

  template <typename T>
  std::vector<CountType> get_random_vector(T size, T min, T max) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(min, max);  // inclusive

    std::vector<T> v(size);
    std::transform(v.cbegin(), v.cend(), v.begin(), [&](auto) { return distr(gen); });

    return v;
  }

  //

  template <typename ElementType>
  void T_fails_validation() {
    boost::mpi::communicator world;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    krylov_m_num_of_alternations_signs_mpi::TestMPITaskParallel<ElementType, CountType> testMpiTaskParallel(
        taskDataPar);

    if (world.rank() == 0) {
      taskDataPar->outputs_count.emplace_back(0);
      EXPECT_FALSE(testMpiTaskParallel.validation());
    } else {
      EXPECT_TRUE(testMpiTaskParallel.validation());
    }
  }
};

// clang-format off
using PrecalcOpts = std::tuple<
  CountType              /* count */, 
  std::vector<CountType> /* shift_indices */,
  CountType              /* num */
>;
// clang-format on
class krylov_m_num_of_alternations_signs_mpi_test_precalc : public krylov_m_num_of_alternations_signs_mpi_test,
                                                            public ::testing::WithParamInterface<PrecalcOpts> {
 protected:
  template <typename ElementType>
  void PT_yields_correct_result() {
    boost::mpi::communicator world;
    const auto &[count, shift_indices, num] = GetParam();

    std::vector<ElementType> in;
    CountType out = 0;
    //
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    run_generic_test<ElementType>(world, count, in, shift_indices, out, taskDataPar);

    if (world.rank() == 0) {
      ASSERT_EQ(out, num);
    }
  }
};

class krylov_m_num_of_alternations_signs_mpi_test_random : public krylov_m_num_of_alternations_signs_mpi_test,
                                                           public ::testing::WithParamInterface<int> {
 protected:
  template <typename ElementType>
  void PT_yields_correct_result_random() {
    boost::mpi::communicator world;
    const auto count = GetParam();

    std::vector<ElementType> in;
    CountType out = 0;
    std::vector<CountType> shift_indices;
    //
    if (world.rank() == 0) {
      const auto shift_indices_count = get_random_vector<CountType>(1, 0, count - 1)[0];
      shift_indices = get_random_vector<CountType>(shift_indices_count, 0, count - 1);
    }
    //
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    run_generic_test<ElementType>(world, count, in, shift_indices, out, taskDataPar);

    if (world.rank() == 0) {
      CountType reference_num = 0;

      //
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>(*taskDataPar);
      taskDataSeq->outputs[0] = reinterpret_cast<uint8_t *>(&reference_num);

      //
      krylov_m_num_of_alternations_signs_mpi::TestMPITaskSequential<ElementType, CountType> testMpiTaskSequential(
          taskDataSeq);
      ASSERT_TRUE(testMpiTaskSequential.validation());
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_EQ(out, reference_num);
    }
  }
};

#define DECL_TYPE_VALUE_PARAMETRIZED_TEST(TypeParam, Fixture, TestName, ...) \
  TEST_P(Fixture, TestName##__##TypeParam) { PT_##TestName<TypeParam>(__VA_ARGS__); }
#define DECL_TYPE_VALUE_PARAMETRIZED_TEST_ALL(Fixture, TestName, ...) \
  T_DEF(DECL_TYPE_VALUE_PARAMETRIZED_TEST, Fixture, TestName, __VA_ARGS__)

#define DECL_TYPE_PARAMETRIZED_TEST(TypeParam, Fixture, TestName, ...) \
  TEST_F(Fixture, TestName##__##TypeParam) { T_##TestName<TypeParam>(__VA_ARGS__); }
#define DECL_TYPE_PARAMETRIZED_TEST_ALL(Fixture, TestName, ...) \
  T_DEF(DECL_TYPE_PARAMETRIZED_TEST, Fixture, TestName, __VA_ARGS__)

INSTANTIATE_TEST_SUITE_P(
    krylov_m_num_of_alternations_signs_mpi_test, krylov_m_num_of_alternations_signs_mpi_test_precalc,
    // clang-format off
    ::testing::Values(
      std::make_tuple(129, std::vector<CountType>{0, 1, /* . */ 3, /* . */ 5, 6, 7, /* . */ 12 /* . */}, 7),
      std::make_tuple(129, std::vector<CountType>{0, /* . */}, 1),
      std::make_tuple(129, std::vector<CountType>{/* . */ 128}, 1),
      std::make_tuple(129, std::vector<CountType>{/* . */ 64 /* . */}, 2),
      std::make_tuple(129, std::vector<CountType>{/* . */ 43, /* . */ 86, /* . */}, 4),
      std::make_tuple(129, std::vector<CountType>{/* . */}, 0),
      std::make_tuple(128, std::vector<CountType>{0, 1, /* . */ 3, /* . */ 5, 6, 7, /* . */ 12 /* . */}, 7),
      std::make_tuple(128, std::vector<CountType>{0, /* . */}, 1),
      std::make_tuple(128, std::vector<CountType>{/* . */ 127}, 1),
      std::make_tuple(128, std::vector<CountType>{/* . */ 64 /* . */}, 2),
      std::make_tuple(129, std::vector<CountType>{/* . */ 43, /* . */ 86, /* . */}, 4),
      std::make_tuple(129, std::vector<CountType>{/* . */ 42, /* . */ 84, /* . */}, 4),
      std::make_tuple(128, std::vector<CountType>{/* . */}, 0),
      std::make_tuple(4,   std::vector<CountType>{/* . */}, 0),
      std::make_tuple(4,   std::vector<CountType>{/* . */ 2 /* . */}, 2),
      std::make_tuple(1,   std::vector<CountType>{/* . */}, 0), 
      std::make_tuple(1,   std::vector<CountType>{0}, 0),
      std::make_tuple(0,   std::vector<CountType>{/* . */}, 0)
    )
    // clang-format on
);

INSTANTIATE_TEST_SUITE_P(krylov_m_num_of_alternations_signs_mpi_test,
                         krylov_m_num_of_alternations_signs_mpi_test_random,
                         ::testing::Values(1, 2, 3, 4, 5, 128, 129));

DECL_TYPE_VALUE_PARAMETRIZED_TEST_ALL(krylov_m_num_of_alternations_signs_mpi_test_precalc, yields_correct_result);
DECL_TYPE_VALUE_PARAMETRIZED_TEST_ALL(krylov_m_num_of_alternations_signs_mpi_test_random, yields_correct_result_random);
DECL_TYPE_PARAMETRIZED_TEST_ALL(krylov_m_num_of_alternations_signs_mpi_test, fails_validation);