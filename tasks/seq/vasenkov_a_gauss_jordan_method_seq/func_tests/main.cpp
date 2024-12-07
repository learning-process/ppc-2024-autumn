#include <gtest/gtest.h>

#include <vector>

#include "seq/vasenkov_a_gauss_jordan_method_seq/include/ops_seq.hpp"


TEST(vasenkov_a_gauss_jordan_method_seq, zero_matrix) {
    std::vector<double> input_matrix = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int n = 3;
    std::vector<double> output_result(n * (n + 1));

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
    taskDataSeq->outputs_count.emplace_back(output_result.size());

    vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
    ASSERT_TRUE(taskSequential.validation());
    taskSequential.pre_processing();
    
    EXPECT_FALSE(taskSequential.run());
    
    taskSequential.post_processing();
}

TEST(vasenkov_a_gauss_jordan_method_seq, identity_matrix) {
    std::vector<double> input_matrix = {1, 0, 0, 0,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0};
    int n = 3;
    std::vector<double> output_result(n * (n + 1));

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
    taskDataSeq->outputs_count.emplace_back(output_result.size());

    vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
    ASSERT_TRUE(taskSequential.validation());
    taskSequential.pre_processing();
    
    ASSERT_TRUE(taskSequential.run());
    
    std::vector<double> expected_result = {1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0};
                                           
    ASSERT_EQ(output_result, expected_result);
}

TEST(vasenkov_a_gauss_jordan_method_seq, non_square_matrix) {
    std::vector<double> input_matrix = {1, 2, -1, -4,
                                         2, 3, 3, -11,
                                         -1, -2, 4, 6};
                                         
    int n = 3;
    std::vector<double> output_result(n * (n + 1));

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
    taskDataSeq->outputs_count.emplace_back(output_result.size());

    vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
    
    ASSERT_TRUE(taskSequential.validation());
    
    taskSequential.pre_processing();
    
    ASSERT_TRUE(taskSequential.run());
    
   std::vector<double> expected_result = {1.0, 0.0, -2.0,
                                          0.0, 1.0, -1.0,
                                          0.0, 0.0, -2.0};
                                          
   ASSERT_EQ(output_result.size(), expected_result.size());
   for (size_t i = 0; i < output_result.size(); ++i) {
       ASSERT_NEAR(output_result[i], expected_result[i], 1e-5);
   }
}

TEST(vasenkov_a_gauss_jordan_method_seq, singular_matrix) {
   std::vector<double> input_matrix = {2, -2, -4,
                                        -4, -6, -10,
                                        -2, -4, -6};
   int n = 3;
   std::vector<double> output_result(n * (n + 1));

   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
   taskDataSeq->inputs_count.emplace_back(input_matrix.size());

   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
   taskDataSeq->inputs_count.emplace_back(1);

   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
   taskDataSeq->outputs_count.emplace_back(output_result.size());

   vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential taskSequential(taskDataSeq);
   
   ASSERT_TRUE(taskSequential.validation());
   
   taskSequential.pre_processing();
   
   EXPECT_FALSE(taskSequential.run());
   
   taskSequential.post_processing();
}
