#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Eigen\Dense"

using namespace std;

int solTRByEigen(double R[][3], double *T, double (*vPtg)[3], double (*vPtl)[3], double *Wi, int nNum);

int main()
{
	int num = 3;

	double(*P)[3] = new double[num][3];
	P[0][0] = -171.164, P[0][1] = -14.6164, P[0][2] = -660.198; // x,y,z
	P[1][0] = -47.77,	P[1][1] = -8.08739, P[1][2] = -689.577;
	P[2][0] = -118.297,	P[2][1] = 51.3259,	P[2][2] = -743.167;

	double(*Q)[3] = new double[num][3];
	Q[0][0] = -37.0991, Q[0][1] = -72.3808, Q[0][2] = -662.87;
	Q[1][0] = 87.7909,	Q[1][1] = -71.975,	Q[1][2] = -686.096;
	Q[2][0] = 22.8645,	Q[2][1] = -10.0038,	Q[2][2] = -743.771;

	double *Wi = new double[num];
	Wi[0] = 0.334814, Wi[1] = 0.298856, Wi[2] = 0.36633;

	double dR[3][3];
	double dT[3];

	int result = solTRByEigen(dR, dT, P, Q, Wi, num);

	delete[] P;
	delete[] Q;
	delete[] Wi;
	
	cout << dR[0][0] << " " << dR[0][1] << " " << dR[0][2] << endl;
	cout << dR[1][0] << " " << dR[1][1] << " " << dR[1][2] << endl;
	cout << dR[2][0] << " " << dR[2][1] << " " << dR[2][2] << endl;
	cout << dT[0] << " " << dT[1] << " " << dT[2] << endl;
	system("pause");

	return 0;
}

int solTRByEigen(double dR[][3], double *dT, double(*vPtp)[3], double(*vPtq)[3], double *Wi, int nNum)
{ 
	using namespace Eigen;
	MatrixXd AvgP, AvgQ;
	AvgP = MatrixXd::Zero(3, 1);
	AvgQ = MatrixXd::Zero(3, 1);
	for (int i = 0; i< nNum; i++)
	{
		AvgP(0, 0) += Wi[i] * vPtp[i][0];
		AvgP(1, 0) += Wi[i] * vPtp[i][1];
		AvgP(2, 0) += Wi[i] * vPtp[i][2];

		AvgQ(0, 0) += Wi[i] * vPtq[i][0];
		AvgQ(1, 0) += Wi[i] * vPtq[i][1];
		AvgQ(2, 0) += Wi[i] * vPtq[i][2];
	}

	MatrixXd P(3, nNum), Q(3, nNum);
	MatrixXd W;
	W = MatrixXd::Zero(nNum, nNum);
	for (int i = 0; i < nNum; i++)
	{
		P(0, i) = vPtp[i][0] - AvgP(0, 0);
		P(1, i) = vPtp[i][1] - AvgP(1, 0);
		P(2, i) = vPtp[i][2] - AvgP(2, 0);

		Q(0, i) = vPtq[i][0] - AvgQ(0, 0);
		Q(1, i) = vPtq[i][1] - AvgQ(1, 0);
		Q(2, i) = vPtq[i][2] - AvgQ(2, 0);

		W(i,i) = Wi[i];
	}

	Matrix3d H, U, V;
	MatrixXd A;
	A = P*W;
	H = A*Q.transpose();

	JacobiSVD<MatrixXd> svd(H, ComputeThinU | ComputeThinV);
	U = svd.matrixU();
	V = svd.matrixV();

	Matrix3d R;
	R = V * U.transpose();
	if (R.determinant() < 0)
	{
		V(0, 0) = -V(0, 0);
		V(1, 0) = -V(1, 0);
		V(2, 0) = -V(2, 0);
	}
	MatrixXd T(3, 1);
	T = (-1)*R*AvgP + AvgQ;

	dR[0][0] = R(0, 0), dR[0][1] = R(0, 1), dR[0][2] = R(0, 2);
	dR[1][0] = R(1, 0), dR[1][1] = R(1, 1), dR[1][2] = R(1, 2);
	dR[2][0] = R(2, 0), dR[2][1] = R(2, 1), dR[2][2] = R(2, 2);
	dT[0] = T(0, 0), dT[1] = T(1, 0), dT[2] = T(2, 0);

	return 0;
}
