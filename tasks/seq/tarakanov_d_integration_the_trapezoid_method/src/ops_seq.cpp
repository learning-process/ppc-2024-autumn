// Copyright 2024 Tarakanov Denis
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

#include <thread>
#include <iostream>

using namespace std::chrono_literals;

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::pre_processing() 
{
  internal_order_test();
  // Init value for input and output
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  h = *reinterpret_cast<double*>(taskData->inputs[2]);
  res = 0;
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::validation() 
{
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::run() 
{
    internal_order_test();

      int n = static_cast<int>((b - a) / h);
    
    // Начальное значение интеграла
    double integral = 0.0;
    
    // Суммируем площади трапеций
    for (int i = 0; i < n; ++i) {
        double x0 = a + i * h;        // Левый конец трапеции
        double x1 = a + (i + 1) * h;  // Правый конец трапеции
        integral += 0.5 * (x0 * x0 + x1 * x1) * h; // Площадь трапеции
    }

    res = integral;
    std::cout << "RES = " << res << "\n\n\n\n";

    return true;
  // internal_order_test();
  // double integral = 0.5 * (f(a) + f(b));
  // for (double x = a + h; x < b; x += h) 
  // {
  //   integral += f(x);
  // }
  // res = integral * h;
  // return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::post_processing() 
{
  internal_order_test();
  taskData->outputs[0] = reinterpret_cast<uint8_t*>(&res);
  std::cout << "outputs[0] = " << *reinterpret_cast<double*>(taskData->outputs[0]) << "\n\n\n\n";
  return true;
}
