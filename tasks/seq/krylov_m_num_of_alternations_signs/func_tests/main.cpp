#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include "seq/krylov_m_num_of_alternations_signs/include/ops_seq.hpp"

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
class krylov_m_num_of_alternations_signs_seq_test : public testing::Test {
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

TYPED_TEST_SUITE(krylov_m_num_of_alternations_signs_seq_test, ElementTypes);

TYPED_TEST(krylov_m_num_of_alternations_signs_seq_test, yields_correct_result) {
  using ElementType = TypeParam;

  for (const auto &params : this->testParamsSet_) {
    const auto &[count, shift_indices, num] = params;

    //
    std::vector<ElementType> in(count);
    CountType out = 0;

    std::iota(in.begin(), in.end(), 1);

    for (auto idx : shift_indices) {
      in[idx] *= -1;
    }

    //
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataSeq->outputs_count.emplace_back(1);

    //
    krylov_m_num_of_alternations_signs_seq::TestTaskSequential<ElementType, CountType> testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(num, out);
  }
}

#define DECL_TEST_PARAMS(TypeParam)                                                                              \
  template <>                                                                                                    \
  std::vector<krylov_m_num_of_alternations_signs_seq_test<TypeParam>::TestParamsT>                               \
      krylov_m_num_of_alternations_signs_seq_test<TypeParam>::testParamsSet_{                                    \
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
