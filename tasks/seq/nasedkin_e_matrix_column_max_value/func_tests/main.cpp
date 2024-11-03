#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include "core/task/include/task.hpp"
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

namespace nasedkin_e_matrix_column_max_value_test {

TEST(SeqMatrixColumnMaxTest, BasicTest) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input = {1, 2, 3, 4, 5};
    int output = 0;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    taskData->outputs_count.emplace_back(1);

    auto task = std::make_shared<nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential>(taskData, "max");

    ASSERT_TRUE(task->validation());
    task->pre_processing();
    task->run();
    task->post_processing();

    ASSERT_EQ(output, 5);
}

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
