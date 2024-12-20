#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/pikarychev_i_sleeping_barber/include/ops_mpi.hpp"

static void test(int capacity) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&capacity));
    taskDataPar->inputs_count.emplace_back(1);
  }

  pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (!testMpiTaskParallel.validation()) {
    GTEST_SKIP();
    return;
  }
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
}

TEST(pikarychev_i_sleeping_barber_mpi_test, test_1) { test(1); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_2) { test(2); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_3) { test(3); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_4) { test(4); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_5) { test(5); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_6) { test(6); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_7) { test(7); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_8) { test(8); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_9) { test(9); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_10) { test(10); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_11) { test(11); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_12) { test(12); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_13) { test(13); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_16) { test(16); }
TEST(pikarychev_i_sleeping_barber_mpi_test, test_17) { test(17); }
TEST(pikarychev_i_sleeping_barber_mpi_test, fails_validation) {
  boost::mpi::communicator world;
  int capacity = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&capacity));
    taskDataPar->inputs_count.emplace_back(1);
  }

  pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.size() < 2) {
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}
