#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

#include "mpi/krylov_m_num_of_alternations_signs/include/ops_mpi.hpp"

#define EXPAND(x) x
#define REMOVE_FIRST(...) EXPAND(REMOVE_FIRST_SUB(__VA_ARGS__))
#define REMOVE_FIRST_SUB(X, ...) __VA_ARGS__

#define T_DEF(macro)     \
  EXPAND(macro(int16_t)) \
  EXPAND(macro(int32_t)) \
  EXPAND(macro(int64_t)) \
  EXPAND(macro(float))

using CountType = uint32_t;
#define X(type) , type
using ElementTypes = ::testing::Types<REMOVE_FIRST(T_DEF(X))>;
#undef X

template <typename T>
class krylov_m_num_of_alternations_signs_mpi_test : public testing::Test {
 public:
  // clang-format off
  using TestParamsT = std::tuple<
    CountType      /* count */, 
    std::vector<T> /* shift_indices */,
    CountType      /* num */
  >;
  // clang-format on
  static std::vector<TestParamsT> testParamsSet_;
};

TYPED_TEST_SUITE(krylov_m_num_of_alternations_signs_mpi_test, ElementTypes);

TYPED_TEST(krylov_m_num_of_alternations_signs_mpi_test, yields_correct_result) {
  using ElementType = TypeParam;

  boost::mpi::communicator world;
  for (const auto &params : this->testParamsSet_) {
    const auto &[count, shift_indices, num] = params;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<ElementType> in;
    CountType out = 0;
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
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();
    //

    if (world.rank() == 0) {
      CountType reference_num = 0;

      //
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>(*taskDataPar);
      taskDataSeq->outputs[0] = reinterpret_cast<uint8_t *>(&reference_num);

      //
      krylov_m_num_of_alternations_signs_mpi::TestMPITaskSequential<ElementType, CountType> testMpiTaskSequential(
          taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      ASSERT_EQ(reference_num, num);
      ASSERT_EQ(out, reference_num);
    }

    world.barrier();
  }
}

#define DECL_TEST_PARAMS(TypeParam)                                                                              \
  template <>                                                                                                    \
  std::vector<krylov_m_num_of_alternations_signs_mpi_test<TypeParam>::TestParamsT>                               \
      krylov_m_num_of_alternations_signs_mpi_test<TypeParam>::testParamsSet_{                                    \
          std::make_tuple(129, std::vector<TypeParam>{0, 1, /* . */ 3, /* . */ 5, 6, 7, /* . */ 12 /* . */}, 7), \
          std::make_tuple(129, std::vector<TypeParam>{0, /* . */}, 1),                                           \
          std::make_tuple(129, std::vector<TypeParam>{/* . */ 128}, 1),                                          \
          std::make_tuple(129, std::vector<TypeParam>{/* . */ 64 /* . */}, 2),                                   \
          std::make_tuple(129, std::vector<TypeParam>{/* . */ 43, /* . */ 86, /* . */}, 4),                      \
          std::make_tuple(129, std::vector<TypeParam>{/* . */}, 0),                                              \
          std::make_tuple(128, std::vector<TypeParam>{0, 1, /* . */ 3, /* . */ 5, 6, 7, /* . */ 12 /* . */}, 7), \
          std::make_tuple(128, std::vector<TypeParam>{0, /* . */}, 1),                                           \
          std::make_tuple(128, std::vector<TypeParam>{/* . */ 127}, 1),                                          \
          std::make_tuple(128, std::vector<TypeParam>{/* . */ 64 /* . */}, 2),                                   \
          std::make_tuple(129, std::vector<TypeParam>{/* . */ 43, /* . */ 86, /* . */}, 4),                      \
          std::make_tuple(129, std::vector<TypeParam>{/* . */ 42, /* . */ 84, /* . */}, 4),                      \
          std::make_tuple(128, std::vector<TypeParam>{/* . */}, 0),                                              \
          std::make_tuple(4, std::vector<TypeParam>{/* . */}, 0),                                                \
          std::make_tuple(4, std::vector<TypeParam>{/* . */ 2 /* . */}, 2),                                      \
          std::make_tuple(1, std::vector<TypeParam>{/* . */}, 0),                                                \
          std::make_tuple(1, std::vector<TypeParam>{0}, 0),                                                      \
          std::make_tuple(0, std::vector<TypeParam>{/* . */}, 0)};

T_DEF(DECL_TEST_PARAMS)
