#include <gtest/gtest.h>

#include <vector>

#include "seq/guseynov_e_marking_comps_of_bin_image/include/ops_seq.hpp"

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_with_isolated_points) {
    const int rows = 3;
    const int columns = 3;
    std::vector<int> in = {0, 1, 1, 1, 1, 0, 0, 1, 1};
    std::vector<int> out(rows * columns, -1);
    std::vector<int> expected_out = {2, 1, 1, 1, 1, 3, 4, 1, 1};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(columns);

    guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_with_no_isolated_object) {
    const int rows = 3;
    const int columns = 3;
    std::vector<int> in = {0, 0, 0, 1, 1, 1, 0, 0, 1};
    std::vector<int> out(rows * columns, -1);
    std::vector<int> expected_out = {2, 2, 2, 1, 1, 1, 3, 3, 1};

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(rows);
    taskDataSeq->outputs_count.emplace_back(columns);

    guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
    ASSERT_EQ(expected_out, out);
}