#include "ops_mpi.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace nasedkin_e_matrix_column_max_value_mpi {

    std::vector<int> FindColumnMaxMPI(const std::vector<std::vector<int>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return {};
        }

        int world_size;
        int world_rank;
        boost::mpi::communicator world;
        world_size = world.size();
        world_rank = world.rank();

        int num_rows = static_cast<int>(matrix.size());
        int num_cols = static_cast<int>(matrix[0].size());
        int local_rows = num_rows / world_size;
        int remainder = num_rows % world_size;

        std::vector<std::vector<int>> local_matrix(local_rows + (world_rank < remainder ? 1 : 0), std::vector<int>(num_cols));
        boost::mpi::scatter(world, matrix.data(), local_matrix.data(), local_matrix.size(), 0);

        std::vector<int> local_max(num_cols, std::numeric_limits<int>::min());
        for (const auto& row : local_matrix) {
            for (size_t col = 0; col < row.size(); ++col) {
                local_max[col] = std::max(local_max[col], row[col]);
            }
        }

        std::vector<int> global_max(num_cols);
        boost::mpi::reduce(world, local_max.data(), global_max.data(), num_cols, std::greater<int>(), 0);

        return global_max;
    }

}
