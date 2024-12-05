#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// Предположим, что SimpleIntMPI уже включён
#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"
#include "core/task/include/task_data.hpp"

// Последовательная реализация 3x3 Гауссовой свёртки для проверки корректности.
// Применяем то же ядро:
// [1 2 1
//  2 4 2
//  1 2 1] / 16
static void gaussian_3x3_seq(const std::vector<int>& input, int width, int height, std::vector<int>* output) {
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    output->resize(width * height);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int sum = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int rr = std::min(std::max(r+kr,0), height-1);
                    int cc = std::min(std::max(c+kc,0), width-1);
                    sum += input[rr*width + cc]*kernel[kr+1][kc+1];
                }
            }
            (*output)[r*width + c] = sum / 16;
        }
    }
}

TEST(anufriev_d_linear_image_func, SmallImageTest) {
    boost::mpi::communicator world;

    int width = 5;
    int height = 4;
    std::vector<int> input(width * height);
    // Заполним простой градиент для наглядности
    std::iota(input.begin(), input.end(), 0);

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    std::vector<int> output(width * height, 0);
    if (world.rank() == 0) {
        // Устанавливаем входы
        taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
        taskData->inputs_count.push_back(input.size());
        // Доп. параметры: width и height - предположим, что сохраняем в inputs_count[1] и [2]
        taskData->inputs_count.push_back(width);
        taskData->inputs_count.push_back(height);

        // Устанавливаем выходы
        taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
        taskData->outputs_count.push_back(output.size());
    }

    auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
    ASSERT_TRUE(task->validation());
    ASSERT_TRUE(task->pre_processing());
    ASSERT_TRUE(task->run());
    ASSERT_TRUE(task->post_processing());

    if (world.rank() == 0) {
        // Сравним с последовательным результатом
        std::vector<int> expected;
        gaussian_3x3_seq(input, width, height, &expected);
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < output.size(); i++) {
            ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
        }
    }
}

TEST(anufriev_d_linear_image_func, LargerImageRandomTest) {
    boost::mpi::communicator world;

    int width = 100;
    int height = 80;

    std::vector<int> input(width * height);
    // Заполним случайными данными
    srand(123);
    for (auto &val : input) {
        val = rand() % 256;
    }

    std::vector<int> output(width * height, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
        taskData->inputs_count.push_back(input.size());
        // Запишем ширину и высоту
        taskData->inputs_count.push_back(width);
        taskData->inputs_count.push_back(height);

        taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
        taskData->outputs_count.push_back(output.size());
    }

    auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
    ASSERT_TRUE(task->validation());
    ASSERT_TRUE(task->pre_processing());
    ASSERT_TRUE(task->run());
    ASSERT_TRUE(task->post_processing());

    if (world.rank() == 0) {
        std::vector<int> expected;
        gaussian_3x3_seq(input, width, height, &expected);
        ASSERT_EQ(output.size(), expected.size());
        for (size_t i = 0; i < output.size(); i++) {
            ASSERT_EQ(output[i], expected[i]) << "Difference at i=" << i;
        }
    }
}