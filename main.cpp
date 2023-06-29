#include <iostream>
#include <typeinfo>
#include "engine.hpp"
#include "utils.hpp"
#include <cstdarg>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <optional>
#include <concepts>
#include <array>

using std::cout, std::endl, std::vector;

/*
TODO: replace all asserts with exceptions or something else
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

class Test {
private:
    int id = 0;
public:
    Test(int x = 0) {id = x; cout << "Def constr " << id << endl;}
    ~Test() {cout << "Destr " << id << endl;}
    Test(Test&& t) noexcept {
        cout << "move constr " << id << endl;
        id = t.id;
        t.id = 0;
    }
    Test(const Test& t){
        cout << "copy constr " << id << endl;
        this->id = t.id;
    }
    Test& operator=(const Test& other){
        cout << "Copy assign " << id << endl;
        return *this;
    }
    Test& operator=(Test&& other){
        cout << "Move assign " << id << endl;
        return *this;
    }
};

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

class Base{
public:
    Base() {cout << "Base constructor\n";}
    void hello() {cout<<"hello from base function\n";}
    ~Base() {cout << "Goodbye from Base destructor\n";}
};

class Derived:public Base{
public:
    Derived() {cout << "Derived constructor\n";}
    void hello()  {cout << "hello from derived function\n";}
    ~Derived() {cout << "Goodbye from Derived destructor\n";}
};

#include <cstring>
class String{
    char* str = nullptr;
public:
    const char* get_string() {return str;}
    String(const char* str_){
        cout << "Constructor\n";
        this->str = new char[strlen(str_) + 1];
        strcpy_s(this->str, strlen(str_) + 1, str_);
        //str[strlen(str_)] = '\0';
    }
    String(String&& other){
        cout << "Move constructor\n";
        this->str = other.str;
        other.str = nullptr;
    }
    String(const String& other){
        cout << "Copy constructor\n";
        this->str = new char[strlen(other.str) + 1];
        strcpy_s(this->str, strlen(other.str) + 1, other.str);
       // str[strlen(other.str)] = '\0';
    }
    ~String(){
        cout << "Destructor\n";
        cout << this->str;
        delete[] this->str;
    }
};

String func(char* s){
    return s;
}




int main() {
//    Tensor<int> t1({2,2,3}, 0), t2(t1);
//    cout << t2 + t1;
    cout << sizeof(Tensor<int>) << endl;
    cout << sizeof(SharedPtr<int>) << "   " << sizeof(Size);
}
