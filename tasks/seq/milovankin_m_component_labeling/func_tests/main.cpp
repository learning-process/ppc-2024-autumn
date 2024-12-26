#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/milovankin_m_component_labeling/include/component_labeling_seq.hpp"

namespace milovankin_m_component_labeling_seq {

static void run_test_seq(std::vector<uint8_t>& image, size_t rows, size_t cols,
                         std::vector<uint32_t>& labels_expected) {
  ASSERT_EQ(rows * cols, labels_expected.size());
  std::vector<uint32_t> labels_actual(labels_expected.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(cols);

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(labels_actual.data()));
  taskDataPar->outputs_count.emplace_back(rows);
  taskDataPar->outputs_count.emplace_back(cols);

  ComponentLabelingSeq componentLabeling(taskDataPar);

  ASSERT_TRUE(componentLabeling.validation());
  ASSERT_TRUE(componentLabeling.pre_processing());

  componentLabeling.run();
  componentLabeling.post_processing();

  ASSERT_EQ(labels_actual, labels_expected);
}

}  // namespace milovankin_m_component_labeling_seq

// clang-format off
TEST(milovankin_m_component_labeling_seq, input_1) {
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

  milovankin_m_component_labeling_seq::run_test_seq(img, 4, 4, expected);
}

TEST(milovankin_m_component_labeling_seq, input_2) {
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

  milovankin_m_component_labeling_seq::run_test_seq(img, 4, 4, expected);
}

TEST(milovankin_m_component_labeling_seq, input_circle) {
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

  milovankin_m_component_labeling_seq::run_test_seq(img, 5, 5, expected);
}

TEST(milovankin_m_component_labeling_seq, input_3) {
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

  milovankin_m_component_labeling_seq::run_test_seq(img, 5, 5, expected);
}

TEST(milovankin_m_component_labeling_seq, input_single_row) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 1};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 2};
  milovankin_m_component_labeling_seq::run_test_seq(img, 1, 6, expected);
}

TEST(milovankin_m_component_labeling_seq, input_single_col) {
  std::vector<uint8_t> img = {1, 1, 0, 1, 1, 1};
  std::vector<uint32_t> expected = {1, 1, 0, 2, 2, 2};
  milovankin_m_component_labeling_seq::run_test_seq(img, 6, 1, expected);
}

TEST(milovankin_m_component_labeling_mpi, input_empty_vector) {
  std::vector<uint8_t> img = {};
  std::vector<uint32_t> expected = {};

  milovankin_m_component_labeling_seq::run_test_seq(img, 0, 0, expected);
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
  milovankin_m_component_labeling_seq::run_test_seq(img, 3, 6, expected);
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

  milovankin_m_component_labeling_seq::run_test_seq(img, 2, 6, expected);
}
// clang-format on
