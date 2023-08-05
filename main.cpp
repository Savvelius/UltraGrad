#include <iostream>
#include <typeinfo>
#include "engine.hpp"
#include "utils.hpp"
#include <cstdarg>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <optional>
#include "timer.hpp"
#include <array>
#include <span>

using std::cout, std::endl, std::vector;

/*
TODO: replace all asserts with exceptions or something else
TODO: in all divisions by zero/logs of zero replace asserts with infinity (use std::numeric_limits<T>::infinity)
*/
template<typename T>
void printVec(vector<T> vec) {
    cout << '[';
	for (auto el : vec) {
		cout << el << ", ";
	}
	cout << ']' << std::endl;
}
template<typename T, uint32_t size>
void printArr(T arr[size]){
    cout << '[';
    for (uint32_t i = 0; i < size; i++) {
        cout << arr[i] << ", ";
    }
    cout << ']' << std::endl;
}

class Matrix {
private:
    std::array<int, 2> shape;
    int* data = nullptr;
public:
    Matrix(std::array<int, 2> shape_, int start_val){
        shape = shape_;
        int prod = shape[0] * shape[1];
        data = new int[prod];
        for (int i = 0; i < prod; i++){
            data[i] = start_val;
            start_val++;
        }
    }
    friend std::ostream& operator << (std::ostream& out, const Matrix& m) {
        out << '[';
        for (int i = 0; i < m.shape[0]; i++){
            out << '[';
            for (int j = 0; j < m.shape[1]; j++){
                out << m.data[i * m.shape[1] + j] << ", ";
            }
            out << "],\n";
        }
        out << ']' << endl;
        return out;
    }
};
class Ptr {
    int* memory = nullptr;
    bool is_owner = true;
public:
    Ptr() = default;
    Ptr(size_t size) {
        memory = new int[size];
    }
    Ptr(const Ptr& other);              // deep copy
    Ptr(Ptr&& other);                   // shallow copy
    Ptr& operator=(const Ptr& other);   // deep copy
    Ptr& operator=(Ptr&& other);        // shallow copy
    ~Ptr();
};

struct Time {
    const len_type time_seq;
    const len_type time_par;
    const float time_seq_mean;
    const float time_par_mean;

    Time(len_type ts, len_type tp, int num)
        :time_seq{ts}, time_par{tp},
         time_seq_mean{float(ts) / num},
         time_par_mean{float(tp) / num} {}

    friend std::ostream& operator << (std::ostream& out, const Time& obj ) {
        out << "Sequential = " << obj.time_seq << " | Mean = " << obj.time_seq_mean << endl <<
               "Parallel = "   << obj.time_par << " | Mean = " << obj.time_par_mean << endl <<
               "Sequential / Parallel = "<< obj.time_seq_mean / obj.time_par_mean << " [10^-6 seconds]" << endl;
        return out;
    }
};

template<typename T>
Time test(int num_iters, const Tensor<T>& t1, const Tensor<T>& t2,
          std::function<Tensor<T>(const Tensor<T>&, const Tensor<T>&)> op) {
    len_type time1 = 0, time2 = 0;
    {
        globals::CPU_MULTITHREAD = true;
        Timer tm(true);
        for (int _ = 0; _ < num_iters; ++_)
            op(t1, t2);
        time1 = tm.get_time();
    }
    {
        globals::CPU_MULTITHREAD = false;
        Timer tm(true);
        for (int _ = 0; _ < num_iters; ++_)
            op(t1, t2);
        time2 = tm.get_time();
    }
    return {time2, time1, num_iters};
}

class A {
    int x;
public:
    A(int y) :x{y} {}
    A get() {
        return A(x);
    }
    int getx() { return x; }
};

class B : public A {
    float y;
public:
    B(int x1, float y1) : A(x1), y{y1} {}
    B(A other) : A(other) {}
    B get() {
        return A::get();
    }
    float gety() { return y; }
};

int main() {
    A a(1);
    B b(1, 2.0);
    cout << b.get().getx();
    auto x = b.get();
//    Tensor<int> t1({10, 4, 136, 55},0),
//                t2({10, 4, 55, 132},0);
//    t1.matmul(t2);
//    cout << test<int>(1, t1, t2, [](const auto& x, const auto& y)
//                                              { return x.matmul(y); });

//    cout << t1.matmul(t2);
//    constexpr int N = 100;
//    constexpr int NUM_ITERS = 30;
//    int A[N][N];
//    int B[N][N];
//    int C[N][N];
//    int time1, time2;
//    auto f = [&A, &B, &C] () {
//        for (int i = 0; i < N; ++i) {
//            for (int j = 0; j < N; ++j) {
//                int temp = 0;
//                for (int k = 0; k < N; ++k) {
//                    temp += A[i][k] * B[k][j];
//                }
//                C[i][j] += temp;
//            }
//        }
//    };
//    {
//        Timer tm("sequential");
//        for (int _ = 0; _ < NUM_ITERS; ++_)
//            f();
//        time1 = tm.get_time();
//    }
//    {
//        Timer tm("parallel");
//        std::thread threads[NUM_ITERS];
//        for (int _ = 0; _ < NUM_ITERS; ++_)
//            threads[_] = std::thread(f);
//        for (int _ = 0; _ < NUM_ITERS; ++_)
//            threads[_].join();
//        time2 = tm.get_time();
//    }
//    cout << "sequential / parallel = " << float(time1) / float(time2);
}
















