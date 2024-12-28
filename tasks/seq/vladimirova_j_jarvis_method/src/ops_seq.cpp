#include "seq/vladimirova_j_jarvis_method/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

namespace vladimirova_j_jarvis_method_seq {
size_t FindMinAngle(vladimirova_j_jarvis_method_seq::Point* A, vladimirova_j_jarvis_method_seq::Point* B,
                    std::vector<vladimirova_j_jarvis_method_seq::Point> vec) {
  size_t min_angle_point = vec.size() + 1;
  double min_angle = 550;
  int reg_x = A->x - (B->x);
  int reg_y = -(A->y - B->y);  // y идет вниз
  for (size_t i = 0; i < vec.size(); i++) {
    vladimirova_j_jarvis_method_seq::Point* C = &vec[i];
    if (C->x < 0) continue;
    if ((A->x == C->x) && (A->y == C->y)) continue;
    if ((B->x == C->x) && (B->y == C->y)) continue;
    int tmp_x = (C->x - B->x);
    int tmp_y = -(C->y - B->y);
    if (reg_x * tmp_y - reg_y * tmp_x <= 0) {
      double BA_length = sqrt(reg_x * reg_x + reg_y * reg_y);
      double BC_length = sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
      double length = BA_length * BC_length;
      double angle = tmp_x * reg_x + tmp_y * reg_y;
      if (length == 0)
        angle = 0;
      else
        angle = angle / (length);
      if (angle < min_angle) {
        min_angle = angle;
        min_angle_point = i;
      }
    }
  }

  return min_angle_point;
}
}  // namespace vladimirova_j_jarvis_method_seq

bool vladimirova_j_jarvis_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<vladimirova_j_jarvis_method_seq::Point>();
  col = (size_t)taskData->inputs_count[1];
  row = (size_t)taskData->inputs_count[0];
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < col * row; i++) {
    if (tmp_ptr[i] != 255) {
      input_.emplace_back((int)(i % col), (int)(i / col));
    }
  }
  res_.clear();
  return true;
}

bool vladimirova_j_jarvis_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  size_t row_i = taskData->inputs_count[0];
  size_t col_i = taskData->inputs_count[1];
  if (row_i <= 1 || col_i <= 1 || taskData->outputs_count[0] <= 0) return false;
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  size_t c = 0;
  int one_row = -1;
  int one_col = -1;
  for (size_t i = 0; i < row_i * col_i; i++) {
    if (tmp_ptr[i] != 255) {
      c++;
      if (one_row == -1)
        one_row = i / row_i;
      else if ((one_row != -2) && (one_row != (int)(i / row_i)))
        one_row = -2;
      if (one_col == -1)
        one_col = i % row_i;
      else if ((one_col != -2) && (one_col != (int)(i % row_i)))
        one_col = -2;
    }
    if ((one_row == -2) && (one_col == -2) && (c > 2)) return true;
  }
  return ((c > 2) && (one_row == -2) && (one_col == -2));
}

bool vladimirova_j_jarvis_method_seq::TestTaskSequential::run() {
  internal_order_test();
  // поиск нижней левой точки
  // x0 y0 x1 y1  x2 y2
  vladimirova_j_jarvis_method_seq::Point* A = &input_[input_.size() - 1];
  // последн¤¤ точка итак сама¤ нижн¤¤, надо найти самую правую

  for (size_t i = input_.size() - 1; i >= 0; i--) {  // y1
    if (A->y != input_[i].y) break;
    A = &input_[i];
  }
  vladimirova_j_jarvis_method_seq::Point* first = A;
  // работа
  // vladimirova_j_jarvis_method_seq::Point tmp = vladimirova_j_jarvis_method_seq::Point((int)row, (int)col);
  vladimirova_j_jarvis_method_seq::Point tmp = vladimirova_j_jarvis_method_seq::Point(-1, A->y);
  // vladimirova_j_jarvis_method_seq::Point * B = &tmp;
  vladimirova_j_jarvis_method_seq::Point* B = A;
  A = &tmp;
  res_ = std::vector<int>();
  do {
    size_t i = vladimirova_j_jarvis_method_seq::FindMinAngle(A, B, input_);
    if (A != first) {
      A->x = -1;
      A->y = -1;
    }
    A = B;
    res_.push_back(A->y);
    res_.push_back(A->x);
    B = &input_[i];

  } while (B != first);

  return true;
}

bool vladimirova_j_jarvis_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  taskData->outputs_count[0] = res_.size();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_data);
  return true;
}
