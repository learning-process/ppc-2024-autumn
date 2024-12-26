#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/milovankin_m_component_labeling/include/component_labeling.hpp"

namespace milovankin_m_component_labeling_mpi {

static void run_test_mpi(std::vector<uint8_t>& image, size_t rows, size_t cols, std::vector<uint32_t>& labels_expected,
                         const std::vector<std::vector<size_t>>& groups_expected) {
  boost::mpi::communicator world;

  ASSERT_EQ(rows * cols, labels_expected.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<uint32_t> labels_actual_par(rows * cols);
  std::vector<uint32_t> labels_actual_seq(rows * cols);

  // Parallel task data
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(labels_actual_par.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(cols);
  }

  // Run parallel
  ComponentLabelingPar componentLabeling(taskDataPar);

  componentLabeling.validation();
  componentLabeling.pre_processing();
  componentLabeling.run();
  componentLabeling.post_processing();

  // Sequential task data
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labels_actual_seq.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(cols);

    // Run sequential
    ComponentLabelingSeq componentLabelingSeq(taskDataSeq);
    ASSERT_TRUE(componentLabelingSeq.validation());
    ASSERT_TRUE(componentLabelingSeq.pre_processing());
    componentLabelingSeq.run();
    componentLabelingSeq.post_processing();

    // Assert results
    ASSERT_EQ(labels_actual_seq, labels_expected);

    // Indecies in the output may vary based on the number of processes used,
    // so instead of comparing the actual output with the expected one,
    // we ensure that output contains the expected components
    for (size_t group = 0; group < groups_expected.size(); ++group) {
      for (size_t idx = 1; idx < groups_expected[group].size(); ++idx) {
        ASSERT_EQ(labels_actual_par[groups_expected[group][idx - 1]], labels_actual_par[groups_expected[group][idx]]);
      }
    }
  }
}
}  // namespace milovankin_m_component_labeling_mpi

// clang-format off
TEST(milovankin_m_component_labeling_mpi, input_1) {
  std::vector<uint8_t> img = {
    0,1,0,1,
    1,1,0,1,
    0,0,0,1,
    1,0,1,0
  };

  std::vector<uint32_t> expected = {
    0,1,0,2,
    1,1,0,2,
    0,0,0,2,
    4,0,5,0
  };

  std::vector<std::vector<size_t>> groups = {
    {1, 4, 5},
    {3, 7, 11},
    {12},
    {14}
  };

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 4, 4, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_2) {
  std::vector<uint8_t> img = {
    1,1,0,1,
    1,1,0,1,
    0,0,0,0,
    1,1,1,1
  };

  std::vector<uint32_t> expected = {
    1,1,0,2,
    1,1,0,2,
    0,0,0,0,
    3,3,3,3
  };

  std::vector<std::vector<size_t>> groups = {
    {0, 1, 4, 5},
    {3, 7},
    {12, 13, 14, 15}
  };

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 4, 4, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_circle) {
  std::vector<uint8_t> img = {
    0,0,0,0,0,
    0,0,1,0,0,
    0,1,0,1,0,
    0,0,1,0,0,
    0,0,0,0,0
  };

  std::vector<uint32_t> expected = {
    0,0,0,0,0,
    0,0,1,0,0,
    0,2,0,1,0,
    0,0,2,0,0,
    0,0,0,0,0
  };

  std::vector<std::vector<size_t>> groups = {
    {7, 13},
    {11, 17}
  };

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 5, 5, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_3) {
  std::vector<uint8_t> img = {
    1,0,0,1,1,
    1,0,0,0,1,
    1,1,1,1,1,
    0,0,0,0,0,
    0,1,1,1,0
  };

  std::vector<uint32_t> expected = {
    1,0,0,1,1,
    1,0,0,0,1,
    1,1,1,1,1,
    0,0,0,0,0,
    0,3,3,3,0
  };

  std::vector<std::vector<size_t>> groups = {
    {0, 3, 4, 5, 9, 10, 11, 12, 13, 14},
    {23, 22, 21}
  };

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 5, 5, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_single_row) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 0};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 0};
  std::vector<std::vector<size_t>> groups = {
    {0, 1},
    {3, 4}
  };
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 1, 6, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_single_col) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 0};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 0};
  std::vector<std::vector<size_t>> groups = {
    {0, 1},
    {3, 4}
  };
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 6, 1, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_empty_vector) {
  std::vector<uint8_t> img = {};
  std::vector<uint32_t> expected = {};
  std::vector<std::vector<size_t>> groups = {
    {}
  };
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 0, 0, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_zero_image) {
  std::vector<uint8_t> img = {
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0
  };
  std::vector<uint32_t> expected = {
    0,0,0,0,0,0,
    0,0,0,0,0,0,
    0,0,0,0,0,0
  };
  std::vector<std::vector<size_t>> groups = {
    {}
  };
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 3, 6, expected, groups);
}

TEST(milovankin_m_component_labeling_mpi, input_ones_image) {
  std::vector<uint8_t> img = {
    1,1,1,1,1,1,
    1,1,1,1,1,1
  };
  std::vector<uint32_t> expected = {
    1,1,1,1,1,1,
    1,1,1,1,1,1
  };
  std::vector<std::vector<size_t>> groups = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  };
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 2, 6, expected, groups);
}

// clang-format on
