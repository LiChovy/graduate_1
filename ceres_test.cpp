#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

struct SkewSymmetricMatrix {
    static Eigen::Matrix3d Compute(const Eigen::Vector3d& epsilon) {
        Eigen::Matrix3d skew;
        skew << 0, -epsilon(2), epsilon(1),
                epsilon(2), 0, -epsilon(0),
                -epsilon(1), epsilon(0), 0;
        return skew;
    }
};

class PlaneCostFunction : public ceres::SizedCostFunction<1, 3> {
public:
    PlaneCostFunction(const Eigen::Vector3d& n, const Eigen::Vector3d& P1, const Eigen::Vector3d& PN)
        : n_(n), P1_(P1), PN_(PN) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        const Eigen::Vector3d epsilon(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Matrix3d skew = SkewSymmetricMatrix::Compute(epsilon);
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        Eigen::Vector3d transformed_P1 = (I + skew) * P1_;
        Eigen::Vector3d transformed_PN = (I + skew) * PN_;

        residuals[0] = n_.dot(transformed_PN - transformed_P1);

        if (jacobians != NULL && jacobians[0] != NULL) {
            Eigen::Matrix3d J;
            for (int i = 0; i < 3; ++i) {
                Eigen::Matrix3d skew_derivative = Eigen::Matrix3d::Zero();
                skew_derivative(i / 3, i % 3) = 1;
                skew_derivative(i % 3, i / 3) = -1;

                Eigen::Vector3d dP1_depsilon = skew_derivative * P1_;
                Eigen::Vector3d dPN_depsilon = skew_derivative * PN_;

                jacobians[0][i] = n_.dot(dPN_depsilon - dP1_depsilon);
            }
        }

        return true;
    }

private:
    Eigen::Vector3d n_, P1_, PN_;
};

int main() {
    std::vector<Eigen::Vector3d> pointsP1, pointsPN, normals;
    std::ifstream file("/home/lizixiao/data/data_1/ptnor.xyz");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x1, y1, z1, xN, yN, zN, nx, ny, nz;
        if (!(iss >> x1 >> y1 >> z1 >> xN >> yN >> zN >> nx >> ny >> nz)) { continue; }
        pointsP1.push_back(Eigen::Vector3d(x1, y1, z1));
        pointsPN.push_back(Eigen::Vector3d(xN, yN, zN));
        normals.push_back(Eigen::Vector3d(nx, ny, nz));
    }
    file.close();

    // Set up the optimization problem
    ceres::Problem problem;
    Eigen::Vector3d epsilon(0, 0, 0);  // Initial guess for epsilon

     for (size_t i = 0; i < pointsP1.size(); ++i) {
        ceres::CostFunction* cost_function = new PlaneCostFunction(normals[i], pointsP1[i], pointsPN[i]);
        problem.AddResidualBlock(cost_function, nullptr, epsilon.data());
    }

    // Solve the problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimized epsilon: " << epsilon.transpose() << std::endl;
    std::cout << summary.FullReport() << std::endl;

    return 0;
}
