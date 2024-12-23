#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/milovankin_m_component_labeling/include/component_labeling.hpp"

namespace milovankin_m_component_labeling_mpi {

static void run_test_mpi(std::vector<uint8_t>& image, size_t rows, size_t cols,
                         std::vector<uint32_t>& labels_expected) {
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
    ASSERT_EQ(labels_actual_seq, labels_actual_par);
    ASSERT_EQ(labels_actual_seq, labels_expected);
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

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 4, 4, expected);
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

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 4, 4, expected);
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

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 5, 5, expected);
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

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 5, 5, expected);
}

TEST(milovankin_m_component_labeling_mpi, input_empty) {
  std::vector<uint8_t> img = {
    0,0,0,
    0,0,0
  };

  std::vector<uint32_t> expected = {
    0,0,0,
    0,0,0
  };

  milovankin_m_component_labeling_mpi::run_test_mpi(img, 2, 3, expected);
}

TEST(milovankin_m_component_labeling_mpi, input_single_row) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 1};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 2};
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 1, 6, expected);
}
TEST(milovankin_m_component_labeling_mpi, input_single_col) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 1};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 2};
  milovankin_m_component_labeling_mpi::run_test_mpi(img, 6, 1, expected);
}
// clang-format on
