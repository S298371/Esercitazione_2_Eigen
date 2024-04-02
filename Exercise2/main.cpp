#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to solve the linear system using PALU decomposition
VectorXd solve_with_palu(MatrixXd A, VectorXd b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

// Function to solve the linear system using QR decomposition
VectorXd solve_with_qr(MatrixXd A, VectorXd b) {
    ColPivHouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}

// Function to calculate the relative error
double relative_error(VectorXd x, VectorXd x_true) {
    return (x - x_true).norm() / x_true.norm();
}

int main() {

    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    VectorXd x_true1(2);
    x_true1 << -1.0, -1.0;
    VectorXd x1_palu, x1_qr;
    double error1_palu, error1_qr;
    x1_palu = solve_with_palu(A1,b1);
    x1_qr= solve_with_qr(A1,b1);
    error1_palu = relative_error(x1_palu,x_true1);
    error1_qr = relative_error(x1_qr,x_true1);
    cout << "System 1:" << endl;
    cout << "PALU Decomposition Solution:\n" << x1_palu << endl;
    cout << "Relative Error (PALU): " << error1_palu << endl;
    cout << "QR Decomposition Solution:\n" << x1_qr << endl;
    cout << "Relative Error (QR): " << error1_qr << endl << endl;


    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    VectorXd x_true2(2);
    x_true2 << -1.0, -1.0;
    VectorXd x2_palu, x2_qr;
    double error2_palu, error2_qr;
    x2_palu = solve_with_palu(A2,b2);
    x2_qr = solve_with_qr(A2,b2);
    error2_palu = relative_error(x2_palu,x_true2);
    error2_qr = relative_error(x2_qr,x_true2);
    cout << "Systemm2:" << endl;
    cout << "PALU Decomposition Solution:\n" <<x2_palu<< endl;
    cout << "Relative Error (PALU): " << error2_palu<<endl;
    cout << "QR Decomposition Solution:\n" <<x2_qr<< endl;
    cout << "Relative Error (QR): " << error2_qr<< endl;



    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    VectorXd x_true3(2);
    x_true3 << -1.0, -1.0;
    VectorXd x3_palu, x3_qr;
    double error3_palu, error3_qr;
    x3_palu = solve_with_palu(A3,b3);
    x3_qr = solve_with_qr(A3,b3);
    error3_palu = relative_error(x3_palu, x_true3);
    error3_qr = relative_error(x3_qr,x_true3);
    cout <<"System 3:" <<endl;
    cout << "PALU decomposition Solution:\n" <<x3_palu<< endl;
    cout << "Relative Error (PALU): " <<error3_palu<< endl;
    cout << "QR Decomposition Solution:\n" << x3_qr<< endl;
    cout << "relative error (QR): " <<error3_qr<< endl;



    return 0;
}

