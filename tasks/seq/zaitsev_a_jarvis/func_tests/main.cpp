#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "seq/zaitsev_a_jarvis/include/ops_seq.hpp"
#include "seq/zaitsev_a_jarvis/include/point.hpp"

using Params =
    std::tuple<std::vector<zaitsev_a_jarvis_seq::Point<double>>, std::vector<zaitsev_a_jarvis_seq::Point<double>>>;

class zaitsev_a_jarvis_seq_test : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(zaitsev_a_jarvis_seq_test, returns_correct_convex_hull) {
  const auto &[points, expected] = GetParam();

  std::vector<zaitsev_a_jarvis_seq::Point<double>> out(points.size(), {0, 0});

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(points.data())));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaitsev_a_jarvis_seq::Jarvis<double> task(taskDataSeq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  out.resize(taskDataSeq->outputs_count[0]);

  ASSERT_EQ(expected.size(), taskDataSeq->outputs_count[0]);
  EXPECT_EQ(out, expected);
}

TEST(zaitsev_a_jarvis_seq_test, validation_fails_on_incorrect_input) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<zaitsev_a_jarvis_seq::Point<double>> points = {};
  std::vector<zaitsev_a_jarvis_seq::Point<double>> out = {};

  taskDataSeq->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(points.data())));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  zaitsev_a_jarvis_seq::Jarvis<double> task(taskDataSeq);
  EXPECT_FALSE(task.validation());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(zaitsev_a_jarvis_seq_test, zaitsev_a_jarvis_seq_test, ::testing::Values(
    // Common case
    Params(
      {{2, 5}, {3, 2}, {4, 5}, {5, 7}, {6, 3}, {9, 7}, {9, 1}, {10, 6}, {11, 4}, {12, 3}}, 
      {{9, 1}, {12, 3}, {10, 6}, {9, 7}, {5, 7}, {2, 5}, {3, 2}}
    ), 
    // Common case
    Params(
      {{0, 0}, {1, 5}, {2, 5}, {5, 1}, {4, 5}, {5, 0}, {5, 5}, {0, 5}}, 
      {{0, 0}, {5, 0}, {5, 5}, {0, 5}}
    ),
    // Common case
    Params(
      {{-13, 8}, {8, -4}, {15, 13}, {-11, -3}, {12, -11}, {0, 6}, {3, -4}, {9, 5}, {-8, 10}, {-9, -5}},
      {{12, -11}, {15, 13}, {-8, 10}, {-13, 8}, {-11, -3}, {-9, -5}}
    ),
    // Common case
    Params(
      {{0.4, 0.5}, {0.1, 0.2}, {0.2, -0.2}, {0.1, -0.5}, {-0.1, -0.1}, {-0.4, 0.4}, {-0.7, 0.6}, {-0.6, -0.4}, {-0.6, -0.7}},
      {{-0.6, -0.7}, {0.1, -0.5}, {0.2, -0.2}, {0.4, 0.5}, {-0.7, 0.6}}
    ),
    // Singleton is the convex hull of itself
    Params(
      {{134, 228}},
      {{134, 228}}
    ),
    // Doubleton is the convex hull of itself
    Params(
      {{103, 105}, {101, 105}},
      {{103, 105}, {101, 105}}
    ),
    // Midpoints of one segments ignored (Collinear points case)
    Params(
      {{-6, -3}, {-4, -2}, {-2, -1}, {0, 0}, {2, 1}, {4, 2}, {6, 3}},
      {{-6, -3}, {6, 3}}
    ),
    // All specific cases in one
    Params(
      {{0, 0}, {1, 1}, {-1, -1}, {-1, 1}, {1, -1}, {2, 2}, {2, 0}, {2, -2}, {-2, 0}, {-2, -2}, {0, -2}, {-2, 2}, {0, 2}},
      {{-2, -2}, {2, -2}, {2, 2}, {-2, 2}}
    )
  )
);

// clang-format on