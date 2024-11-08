#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "../include/ops_seq.hpp"

#define EXPAND(x) x

#define T_DEF(macro, ...)             \
  EXPAND(macro(int16_t, __VA_ARGS__)) \
  EXPAND(macro(int32_t, __VA_ARGS__)) \
  EXPAND(macro(int64_t, __VA_ARGS__)) \
  EXPAND(macro(float, __VA_ARGS__))

using CountType = uint32_t;

// clang-format off
using PredefParam = std::tuple<
  CountType              /* count */, 
  std::vector<CountType> /* shift_indices */,
  CountType              /* num */
>;
// clang-format on

class krylov_m_num_of_alternations_signs_seq_test : public ::testing::TestWithParam<PredefParam> {
 protected:
  template <typename ElementType>
  void PT_yields_correct_result() {
    const auto &[count, shift_indices, num] = GetParam();

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
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(num, out);
  }

  template <typename ElementType>
  void T_fails_validation() {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->outputs_count.emplace_back(0);

    krylov_m_num_of_alternations_signs_seq::TestTaskSequential<ElementType, CountType> testTaskSequential(taskDataSeq);
    EXPECT_FALSE(testTaskSequential.validation());
  }
};

#define DECL_TYPE_VALUE_PARAMETRIZED_TEST(TypeParam, TestName) \
  TEST_P(krylov_m_num_of_alternations_signs_seq_test, TestName##__##TypeParam) { TestName<TypeParam>(); }
#define DECL_TYPE_VALUE_PARAMETRIZED_TEST_ALL(TestName) T_DEF(DECL_TYPE_PARAMETRIZED_TEST, PT_##TestName)

#define DECL_TYPE_PARAMETRIZED_TEST(TypeParam, TestName) \
  TEST_P(krylov_m_num_of_alternations_signs_seq_test, TestName##__##TypeParam) { TestName<TypeParam>(); }
#define DECL_TYPE_PARAMETRIZED_TEST_ALL(TestName) T_DEF(DECL_TYPE_PARAMETRIZED_TEST, T_##TestName)

INSTANTIATE_TEST_SUITE_P(
    krylov_m_num_of_alternations_signs_seq_test, krylov_m_num_of_alternations_signs_seq_test,
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

DECL_TYPE_VALUE_PARAMETRIZED_TEST_ALL(yields_correct_result);
DECL_TYPE_PARAMETRIZED_TEST_ALL(fails_validation);
