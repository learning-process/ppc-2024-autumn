#include <gtest/gtest.h>

#include "mpi/polikanov_v_gradient_method/include/ops_mpi.hpp"

namespace polikanov_v_gradient_method_mpi {

void template_func_test_run(int size, std::vector<double> flat_matrix, std::vector<double> rhs,
                            std::vector<double> initialGuess, std::vector<double> expected, double tolerance = 1e-6) {
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(flat_matrix.data()));
  task->inputs_count.emplace_back(flat_matrix.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
  task->inputs_count.emplace_back(rhs.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(initialGuess.data()));
  task->inputs_count.emplace_back(initialGuess.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  GradientMethod GradientMethod(task);
  ASSERT_TRUE(GradientMethod.validation());
  GradientMethod.pre_processing();
  GradientMethod.run();
  GradientMethod.post_processing();
  for (size_t i = 0; i < expected.size(); i++) ASSERT_NEAR(result[i], expected[i], tolerance);
}

}  // namespace polikanov_v_gradient_method_mpi

TEST(polikanov_v_gradient_method_mpi, test_simple_system) {
  int size = 2;
  std::vector<double> flat_matrix = {4, 1, 1, 3};
  std::vector<double> rhs = {1, 2};
  std::vector<double> initialGuess = {2, 1};
  std::vector<double> expected = {0.09090909, 0.63636364};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_larger_system) {
  int size = 3;
  std::vector<double> flat_matrix = {10, 2, 3, 2, 8, 1, 3, 1, 7};
  std::vector<double> rhs = {7, -4, 6};
  std::vector<double> initialGuess = {0, 0, 0};
  std::vector<double> expected = {0.64285714, -0.74675325, 0.68831169};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_diagonal_matrix) {
  int size = 3;
  std::vector<double> flat_matrix = {5, 0, 0, 0, 3, 0, 0, 0, 2};
  std::vector<double> rhs = {10, 9, 4};
  std::vector<double> initialGuess = {0, 0, 0};
  std::vector<double> expected = {2.0, 3.0, 2.0};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_positive_definite) {
  int size = 3;
  std::vector<double> flat_matrix = {6, 2, 1, 2, 5, 2, 1, 2, 4};
  std::vector<double> rhs = {12, 15, 10};
  std::vector<double> initialGuess = {1, 1, 1};
  std::vector<double> expected = {1.10843373, 2.08433735, 1.18072289};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_with_size_5) {
  int size = 5;
  std::vector<double> flat_matrix = {
      5.128292366630172,  0.5021687792403899,  0.4796975230417943, 0.6452286117540835,  0.6673325787548488,
      0.5021687792403899, 5.328031422170855,   0.4801681858566959, 0.35114819394835335, 0.45168341046221503,
      0.4796975230417943, 0.4801681858566959,  5.117150661826597,  0.5724346627185038,  0.4183351013890984,
      0.6452286117540835, 0.35114819394835335, 0.5724346627185038, 5.687098264497823,   0.5529172614911933,
      0.6673325787548488, 0.45168341046221503, 0.4183351013890984, 0.5529172614911933,  5.554135235876474};
  std::vector<double> rhs = {0.25167592358692537, 0.21906251776090102, 0.6879963644535482, 0.7442739025532313,
                             0.8913147428182813};
  std::vector<double> initialGuess = {0.05, 0.05, 0.1, 0.1, 0.15};
  std::vector<double> expected = {0.00625289376423095, 0.011869990033124153, 0.10957629160768877, 0.10478015154369959,
                                  0.14007691728181526};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, another_test_with_size_5) {
  int size = 5;
  std::vector<double> flat_matrix = {
      5.281908934824547,   0.5080260469785666,  0.29430136279575805, 0.630281468333876,  0.5724326059290666,
      0.5080260469785666,  5.823940254694808,   0.46594730160660087, 0.3698530675086439, 0.40322774937227906,
      0.29430136279575805, 0.46594730160660087, 5.128291179916338,   0.4884163382882024, 0.6214222722098664,
      0.630281468333876,   0.3698530675086439,  0.4884163382882024,  5.53460131980437,   0.6182858381613174,
      0.5724326059290666,  0.40322774937227906, 0.6214222722098664,  0.6182858381613174, 5.454831168899086};
  std::vector<double> rhs = {0.0015958723013507203, 0.054445024863126634, 0.3946797587037579, 0.2746902314303173,
                             0.9657498054979261};
  std::vector<double> initialGuess = {0.01, 0.05, 0.1, 0.1, 0.2};
  std::vector<double> expected = {-0.02407393265206433, -0.006630677829699074, 0.0555397591404793, 0.028872076341333014,
                                  0.1704616173745722};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_with_size_4) {
  int size = 4;
  std::vector<double> flat_matrix = {4.42612301767271,   0.2946444057105056, 0.5199487486942651, 0.6764201253743078,
                                     0.2946444057105056, 4.308736107314684,  0.3509989593044634, 0.4938587844175193,
                                     0.5199487486942651, 0.3509989593044634, 4.951998469214098,  0.3749758390037638,
                                     0.6764201253743078, 0.4938587844175193, 0.3749758390037638, 4.844706032625996};
  std::vector<double> rhs = {0.8064063309162165, 0.9906140241067589, 0.4591795286547167, 0.23861633424918348};
  std::vector<double> initialGuess = {0.2, 0.2, 0.1, 0.05};
  std::vector<double> expected = {0.16077900876475582, 0.21393780014343927, 0.06065785202432189, 0.0003017491873278216};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, another_test_with_size_4) {
  int size = 4;
  std::vector<double> flat_matrix = {4.853305300786797,   0.41452318850352615, 0.34067141874096574, 0.48057099393000374,
                                     0.41452318850352615, 4.544476838178948,   0.4447595168345374,  0.3493420297068756,
                                     0.34067141874096574, 0.4447595168345374,  4.448146576936115,   0.579593777616498,
                                     0.48057099393000374, 0.3493420297068756,  0.579593777616498,   4.909784839634118};
  std::vector<double> rhs = {0.07575195408641833, 0.07495793979072551, 0.3700558287496759, 0.48650723136151446};
  std::vector<double> initialGuess = {0.01, 0.02, 0.05, 0.1};
  std::vector<double> expected = {0.0014607860458476277, 0.002458770461937407, 0.0710586271698828, 0.09038300952821808};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}

TEST(polikanov_v_gradient_method_mpi, test_first_iter_solution) {
  int size = 2;
  std::vector<double> flat_matrix = {4, 1, 1, 3};
  std::vector<double> rhs = {1, 2};
  std::vector<double> initialGuess = {0.09090909, 0.63636364};
  std::vector<double> expected = {0.09090909, 0.63636364};
  double tolerance = 1e-6;
  polikanov_v_gradient_method_mpi::template_func_test_run(size, flat_matrix, rhs, initialGuess, expected, tolerance);
}
