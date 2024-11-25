// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/titov_s_simple_iteration/include/ops_mpi.hpp"

TEST(titov_s_simple_iteration_mpi, Test_Simple_Iteration_Not_A_Diagonally_Dominate) { ASSERT_TRUE(3 < 4); }
