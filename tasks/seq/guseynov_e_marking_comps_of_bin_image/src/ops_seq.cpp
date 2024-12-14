#include "seq/guseynov_e_marking_comps_of_bin_image/include/ops_seq.hpp"

#include <unordered_map>

bool guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential::pre_processing() {
    internal_order_test();
    
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[0];
    int pixels_count = rows * columns;
    image_ = std::vector<int>(pixels_count);
    auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + pixels_count, image_.begin());
    
    labeled_image = std::vector<int>(rows * columns, 1);
    return true;
}

bool guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential::validation() {
    internal_order_test();

    int tmp_rows = taskData->inputs_count[0];
    int tmp_columns = taskData->inputs_count[1];

    for (int x = 0; x < tmp_rows; x++){
        for (int y = 0; y < tmp_columns; y++){
            int pixel = static_cast<int>(taskData->inputs[0][x*tmp_rows + y]);
            if (pixel < 0 || pixel > 1){
                return false;
            }
        }
    }
    return tmp_rows > 0 && tmp_columns > 0 && taskData->outputs_count[0] == tmp_rows * tmp_columns; 
}

bool guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential::run() {
    internal_order_test();

    std::vector<int> label_equivalence;
    int current_label = 2;
    
    for (int x = 0; x < rows; x++){
        for (int y = 0; y < columns; y++){
            int position = x * rows + y;
            if (image_[position] == 0 && labeled_image[position] == 1){

            }
        }
    }
}

bool guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential::post_processing() {
    internal_order_test();
}