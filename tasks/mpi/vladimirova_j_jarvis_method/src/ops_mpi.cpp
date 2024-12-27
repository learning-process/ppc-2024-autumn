#include "mpi/vladimirova_j_jarvis_method/include/ops_mpi.hpp"

#include <cmath>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using namespace vladimirova_j_jarvis_method_mpi;

namespace vladimirova_j_jarvis_method_mpi {
double getAngle(int reg_x, int reg_y, Point B, Point C) {
  if (C.x < 0) return 1000;
  if ((B.x == C.x) && (B.y == C.y)) return 1000;
  int tmp_x = (C.x - B.x);
  int tmp_y = -(C.y - B.y);
  if (reg_x * tmp_y - reg_y * tmp_x > 0) return 1000;
  double BA_length = sqrt(reg_x * reg_x + reg_y * reg_y);
  double BC_length = sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
  double length = BA_length * BC_length;
  double angle = tmp_x * reg_x + tmp_y * reg_y;
  if (length == 0)
    angle = 0;
  else
    angle = angle / (length);
  return angle;
}
std::vector<double> getMinAngleMPI(int reg_x, int reg_y, Point B, std::vector<Point>& vec, bool f) {
  size_t min_angle_point = vec.size() + 1;
  double min_angle = 550;
  size_t sz = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (f && (vec[i] == B)) {
      vec[i].x = -1;
      vec[i].y = -1;
      continue;
    }
    if (vec[i].x < 0) continue;
    double angle = getAngle(reg_x, reg_y, B, vec[i]);
    if (angle < min_angle) {
      min_angle = angle;
      min_angle_point = i;
    }
    sz++;
  }
  return {(double)min_angle_point, min_angle, (double)sz};
}

size_t FindMinAngle(Point* A, Point* B, std::vector<Point> vec) {
  size_t min_angle_point = vec.size() + 1;
  double min_angle = 550;
  int reg_x = A->x - (B->x);
  int reg_y = -(A->y - B->y);  // y идет вниз
  for (size_t i = 0; i < vec.size(); i++) {
    Point* C = &vec[i];
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
}  // namespace vladimirova_j_jarvis_method_mpi

bool vladimirova_j_jarvis_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<Point>();
  col = (size_t)taskData->inputs_count[1];
  row = (size_t)taskData->inputs_count[0];
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < col * row; i++) {
    if (tmp_ptr[i] != 255) {
      input_.emplace_back((int)(i % col), (int)(i / col));
    }
  }
  return true;
}

bool vladimirova_j_jarvis_method_mpi::TestMPITaskSequential::validation() {
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

bool vladimirova_j_jarvis_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  // поиск нижней левой точки
  // x0 y0 x1 y1  x2 y2
  Point* A = &input_[input_.size() - 1];
  // последняя точка итак самая нижняя, надо найти самую правую

  for (size_t i = input_.size() - 1; i >= 0; i--) {  // y1
    if (A->y != input_[i].y) break;
    A = &input_[i];
  }
  Point* first = A;
  // работа
  // vladimirova_j_jarvis_method_seq::Point tmp = vladimirova_j_jarvis_method_seq::Point((int)row, (int)col);
  Point tmp = Point(-1, A->y);
  // vladimirova_j_jarvis_method_seq::Point * B = &tmp;
  Point* B = A;
  A = &tmp;
  res_ = std::vector<int>();
  do {
    size_t i = FindMinAngle(A, B, input_);
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

bool vladimirova_j_jarvis_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  taskData->outputs_count[0] = res_.size();
  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_data);
  return true;
}

bool vladimirova_j_jarvis_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    col = (size_t)taskData->inputs_count[1];
    row = (size_t)taskData->inputs_count[0];
  }
  broadcast(world, col, 0);
  broadcast(world, row, 0);
  if (world.rank() != 0) return true;

  input_ = std::vector<int>();

  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < col * row; i++) {
    if (tmp_ptr[i] != 255) {
      input_.push_back((int)(i % col));
      input_.push_back((int)(i / col));
    }
  }
  res_.clear();
  return true;
}

bool vladimirova_j_jarvis_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
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
  return true;
}

bool vladimirova_j_jarvis_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int delta = ((input_.size() / 2) / world.size()) * 2;
  int ost_point = (input_.size() - delta * world.size()) / 2;
  broadcast(world, delta, 0);
  broadcast(world, ost_point, 0);
  if (rank == 0) {
    if (world.size() != 1) {
      for (int i = 1; i < ost_point; i++) {
        world.send(i, 0, input_.data() + i * (delta) + i * 2, delta + 2);
      }
      for (int i = ost_point + (int)(ost_point == 0); i < world.size(); i++) {
        // std::cout << "\n\n\n PROC " << rank << " "<< i<<" "<<ost_point<<" " << delta + 2 << std::endl;
        int sdvig = ost_point * 2;
        world.send(i, 0, input_.data() + i * (delta) + sdvig, delta);
      }
      if (ost_point > 0) delta += 2;
    }
  } else {
    delta += 2 * (int)(rank < ost_point);
    input_ = std::vector<int>(delta);
    // std::cout << "\n\n\n PROC " << rank<<delta << std::endl;
    world.recv(0, 0, input_.data(), delta);
  }

  local_input_ = std::vector<Point>(delta / 2);

  for (int i = 0; i < delta; i += 2) {
    local_input_[i / 2] = (Point(input_[i], input_[i + 1]));
  }
  // распределение данных

  Point B = Point(-1, -1);
  Point C = Point(-1, -1);
  int reg_x;
  int reg_y;
  int active = 1;
  size_t sz = local_input_.size();
  bool f = false;
  if (rank == 0) {
    B = Point(input_[input_.size() - 2], input_[input_.size() - 1]);
    Point A = Point(-1, -1);
    int j = input_.size() - 3;
    while (input_[j] == B.y) {
      B.x = input_[j - 1];
      j -= 2;
    }
    Point first = B;

    reg_x = -1;
    reg_y = 0;
    // send data reg_x reg_y Bx, By, st;
    std::vector<int> send_data = {reg_x, reg_y, B.x, B.y, active};
    std::vector<double> send0_data(3);
    do {
      double min_angle = 1000;
      send_data = {reg_x, reg_y, B.x, B.y, 1};
      for (int i = 1; i < world.size(); i++) world.send(i, 0, send_data.data(), 5);
      // cвоя часть
      std::vector<double> ans = getMinAngleMPI(reg_x, reg_y, B, local_input_, f);
      C = local_input_[(size_t)ans[0]];
      min_angle = ans[1];
      sz = (size_t)ans[2];
      for (int i = 1; i < world.size(); i++) {
        world.recv(i, 0, send0_data.data(), 3);
        if (send0_data[0] < min_angle) {
          C.x = (int)send0_data[1];
          C.y = (int)send0_data[2];
        }
      }

      res_.push_back(B.y);
      res_.push_back(B.x);
      A = B;
      B = C;
      reg_x = A.x - (B.x);
      reg_y = -(A.y - B.y);  // y идет вниз
      f = true;
    } while (first != B);
    send_data[4] = 0;
    for (int i = 1; i < world.size(); i++) world.send(i, 0, send_data.data(), 5);
  }

  if (rank != 0) {
    std::vector<int> send_data(5);
    std::vector<double> send0_data(3);
    while (active == 1) {
      world.recv(0, 0, send_data.data(), 5);
      active = send_data[4];
      if (active != 1) return true;
      if (sz <= 0) {
        send0_data = {10000, -1, -1};
        world.send(0, 0, send0_data.data(), 3);
        continue;
      }
      B.x = send_data[2];
      B.y = send_data[3];
      double min_angle = 5000;
      std::vector<double> ans = getMinAngleMPI(send_data[0], send_data[1], B, local_input_, f);
      f = true;
      size_t itr = ans[0];
      C = local_input_[itr];
      min_angle = ans[1];
      sz = (size_t)ans[2];
      send0_data = {min_angle, (double)C.x, (double)C.y};
      world.send(0, 0, send0_data.data(), 3);
    }
  }

  return true;
}

bool vladimirova_j_jarvis_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs_count[0] = res_.size();
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_data);
    res_.clear();
    return true;
  }
  return true;
}
