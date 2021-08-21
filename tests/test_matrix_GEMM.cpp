#include <iostream>
#include <cassert>
#include <chrono>
using namespace std;

#include "Matrix.h"

/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-4)
{
	return fabs(a - b) < tolerancis;
}
void test(bool bTest, const string& sMessage = "")
{
	if (bTest) return;

	cout << "Test failed: " << sMessage << endl;
	exit(-1);
}
////////////////////////////////////////////////////////
void disp(const MatrixFloat& m)
{
    cout << "rows=" << m.rows() << " columns=" << m.cols() << endl;
    for( int r=0;r<m.rows();r++)
    {
        for( int c=0;c<m.cols();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
////////////////////////////////////////////////////////
void GEMM_col_row_k_ptr(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.resize(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	vector<float> oneCol(kMax);

	for (Index c = 0; c < cols; c++)
	{
		//unstride data
		for (Index k = 0; k < kMax; k++)
			oneCol[k] = b(k, c);

		for (Index r = 0; r < rows; r++)
		{
			float temp = 0.f;
			const float * pA = a.row(r).data();
			const float * pB = oneCol.data();

			for (Index k = 0; k < kMax; k++)
				temp += (*pA++)*(*pB++);

			ab(r, c) = temp;
		}
	}
}
////////////////////////////////////////////////////////
void GEMM_row_col_k(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax= a.cols();

	for (Index r = 0; r < rows; r++)
		for (Index c = 0; c < cols; c++)
			for (Index k = 0; k < kMax; k++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_row_col_k_block44(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();
	Index rowsb = rows >> 2;
	Index colsb = cols >> 2;
	Index kMaxb = kMax >> 2;

	for (Index rb = 0; rb < rows; rb+=4)
		for (Index cb = 0; cb < cols; cb+=4)
			for(Index kb = 0; kb < kMax; kb+=4)
			{
				//read the a block
				float a_block[4*4];
				for (Index i = 0; i < 4; i++)
					for (Index j = 0; j < 4; j++)
						a_block[i * 4 + j] = a(rb + i, kb + j);

				// read the b block
				float b_block[4*4];
				for (Index i = 0; i < 4; i++)
					for (Index j = 0; j < 4; j++)
						b_block[i * 4 + j] = b(kb + i, cb + j);

				// compute the 4x4 product
				float ab_block[4*4];
				for (Index i = 0; i < 4; i++)
					for (Index j = 0; j < 4; j++)
					{
						float sum = 0.f;
						for (Index k = 0; k < 4; k++)
							sum += a_block[i * 4 + k] * b_block[j + k * 4];
						ab_block[i * 4 + j] = sum;
					}
												
				// add to ab
				for (Index i = 0; i < 4; i++)
					for (Index j = 0; j < 4; j++)
						ab(rb + i, cb + j) += ab_block[i * 4 + j];
			}
}
////////////////////////////////////////////////////////
void GEMM_row_k_col(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index r = 0; r < rows; r++)
		for (Index k = 0; k < kMax; k++)
			for (Index c = 0; c < cols; c++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_col_row_k(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index c = 0; c < cols; c++)
		for (Index r = 0; r < rows; r++)
			for (Index k = 0; k < kMax; k++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_col_k_row(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index c = 0; c < cols; c++)
		for (Index k = 0; k < kMax; k++)
			for (Index r = 0; r < rows; r++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_k_row_col(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index k = 0; k < kMax; k++)
		for (Index r = 0; r < rows; r++)
			for (Index c = 0; c < cols; c++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_k_col_row(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index k = 0; k < kMax; k++)
		for (Index c = 0; c < cols; c++)
			for (Index r = 0; r < rows; r++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void test_GEMM_order()
{
	cout << endl << "GEMM order test:" << endl;

	chrono::steady_clock::time_point start, end;

	for (int sz = 8; sz <= 1024; sz *= 2)
	{
		MatrixFloat m1;
		m1.setRandom(sz, sz);

		MatrixFloat m2;
		m2.setRandom(sz, sz);

		start = chrono::steady_clock::now();
		MatrixFloat m3optim = m1 * m2;
		end = chrono::steady_clock::now();
		long long deltaOptim = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//		cout << m3optim << endl << endl;

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive1;
		GEMM_row_col_k(m1, m2, m3Naive1);
		end = chrono::steady_clock::now();
		long long delta_row_col_k = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive1 = (m3optim - m3Naive1).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive1, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive2;
		GEMM_row_k_col(m1, m2, m3Naive2);
		end = chrono::steady_clock::now();
		long long delta_row_k_col = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive2 = (m3optim - m3Naive2).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive2, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive3;
		GEMM_col_row_k(m1, m2, m3Naive3);
		end = chrono::steady_clock::now();
		long long delta_col_row_k = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive3 = (m3optim - m3Naive3).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive3, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive4;
		GEMM_col_k_row(m1, m2, m3Naive4);
		end = chrono::steady_clock::now();
		long long delta_col_k_row = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive4 = (m3optim - m3Naive4).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive4, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive5;
		GEMM_k_row_col(m1, m2, m3Naive5);
		end = chrono::steady_clock::now();
		long long delta_k_row_col = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive5 = (m3optim - m3Naive5).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive5, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive6;
		GEMM_k_col_row(m1, m2, m3Naive6);
		end = chrono::steady_clock::now();
		long long delta_k_col_row = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive6 = (m3optim - m3Naive6).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive6, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive7;
		GEMM_col_row_k_ptr(m1, m2, m3Naive7);
		end = chrono::steady_clock::now();
		long long delta_col_row_k_ptr = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive7 = (m3optim - m3Naive7).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive7, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive8;
		GEMM_row_col_k_block44(m1, m2, m3Naive8);
	//	cout << m3Naive8 << endl;
		end = chrono::steady_clock::now();
		long long delta_row_col_k_block44 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive8 = (m3optim - m3Naive8).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive8, 0.));

		cout << "Optim/RCK/RKC/CRK/CKR/KRC/KCR/CRKptr/RCK_B44/ size:" << sz << " time: "
			<< deltaOptim << "/"
			<< delta_row_col_k << "/"
			<< delta_row_k_col << "/"
			<< delta_col_row_k << "/"
			<< delta_col_k_row << "/"
			<< delta_k_row_col << "/"
			<< delta_k_col_row << "/"
			<< delta_col_row_k_ptr << "/"
			<< delta_row_col_k_block44 << "/"
			<< endl;
	}
}
////////////////////////////////////////////////////////

int main()
{
	test_GEMM_order();

    cout << "Tests finished." << endl;
    return 0;
}
////////////////////////////////////////////////////////