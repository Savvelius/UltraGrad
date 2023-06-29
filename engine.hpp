#pragma once
#include "size.hpp"
#include "smartptr.hpp"
#include <memory>
#include <cassert>
#include <initializer_list>
#include <ostream>
#include <array>
#include <functional>

template<typename T>
class Tensor {
public:
	Tensor() = default;
	Tensor(std::initializer_list<len_type>, T);
	Tensor(const Tensor<T>&);
	Tensor(Tensor<T>&&) noexcept;
    Tensor(len_type);

	[[nodiscard]] Size     shape() const;
    [[nodiscard]] dim_type dims()  const;
    [[nodiscard]] len_type numel(dim_type start_dim = 0) const;
    [[nodiscard]] len_type getOffset(std::initializer_list<idx_type>) const;

	Tensor<T>& operator = (const Tensor&);
	Tensor<T>& operator = (Tensor&&) noexcept;

    Tensor<T> operator [] (idx_type);       // Ready
	Tensor<T> operator [] (std::initializer_list<idx_type>);    // no multiple arguments until c++23
    Tensor<T> operator [] (const Tensor<bool>&) = delete;       // THINK: maybe useless(move version is more popular)
    Tensor<T> operator [] (Tensor<bool>&&)      = delete;       // will be implemented after boolean ops
    [[deprecated("will be deleted")]] Tensor<T> operator [] (Range<len_type>&&);// indexing array without creating new Range instance

    Tensor<bool> operator == (const Tensor<T>&) = delete;
    Tensor<bool> operator == (T)                = delete;
    Tensor<bool> operator > (const Tensor<T>&)  = delete;
    Tensor<bool> operator > (T)                 = delete;
    Tensor<bool> operator < (const Tensor<T>&)  = delete;
    Tensor<bool> operator < (T)                 = delete;

    Tensor<T>  bin_op(const Tensor<T>&, std::function<T(T, T)>);
    Tensor<T>& bin_op_ip(const Tensor<T>&, std::function<void(T&, T)>);
    Tensor<T>  un_op(std::function<T(T)>);
    Tensor<T>& un_op_ip(std::function<void(T&)>);

    Tensor<T> operator + (const Tensor<T>&);        // continue from here
    Tensor<T> operator + (T)                 = delete;
    Tensor<T> operator - (const Tensor<T>&)  = delete;
    Tensor<T> operator - (T)                 = delete;
    Tensor<T> operator * (const Tensor<T>&)  = delete;
    Tensor<T> operator * (T)                 = delete;
    Tensor<T> operator / (const Tensor<T>&)  = delete;
    Tensor<T> operator / (T)                 = delete;

    Tensor<T>& operator += (const Tensor<T>&);
    Tensor<T>& operator += (T)                 = delete;
    Tensor<T>& operator -= (const Tensor<T>&)  = delete;
    Tensor<T>& operator -= (T)                 = delete;
    Tensor<T>& operator *= (const Tensor<T>&)  = delete;
    Tensor<T>& operator *= (T)                 = delete;
    Tensor<T>& operator /= (const Tensor<T>&)  = delete;
    Tensor<T>& operator /= (T)                 = delete;

    Tensor<T> relu()                = delete;
    Tensor<T> exp()                 = delete;
    Tensor<T> tanh()                = delete;
    Tensor<T> bmm(const Tensor<T>&) = delete;   // batch matrix multiplication

    void backward() = delete;

	~Tensor() = default;

    template<typename U>
    friend std::ostream& operator << (std::ostream&, const Tensor<U>&);
private:
	SharedPtr<T> data;  // FIXME check for compatibility
    Size size;
    // place for a lambda _backward function
    // place for another SharedPtr<T> holding gradient values
	Tensor(SharedPtr<T>&&, Size&&);
};

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T> & other) {
    this->bin_op_ip(other, [](T& mut, T arg) -> void{ mut += arg; });
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::un_op(std::function<T(T)> operation) {
    Tensor<T> out(this->numel());
    out.size = size;
    for (len_type i = 0; i < this->numel(); ++i)
        out.data[i] = operation(data[i]);
    return std::move(out);
}

template<typename T>
Tensor<T> Tensor<T>::bin_op(const Tensor<T> & other, std::function<T(T, T)> operation) {
    Tensor<T> out(this->numel(0));
    if (other.size == size){    // basic case
        out.size = size;
        for (len_type i = 0; i < numel(); i++){
            out.data[i] = operation(data[i], other.data[i]);
        }
    }
    else { // tensor with min size will be added with given stride
        if (size.dims() < other.size.dims()) {

        } else if (size.dims() == other.size.dims()){

        }
    }
    return std::move(out);
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> & other) {
    return this->bin_op(other, [](T x, T y) -> T { return x + y; });
}

template<typename T>
Tensor<T>::Tensor(len_type length) {
    this->data.reserve(length);
}

// FIXME: implement me
template<typename T>
[[deprecated("too complicated for now")]] Tensor<T> Tensor<T>::operator[](Range<len_type> && index) {
    return *this;
    auto info = index.get_info();
    assert(info.start >= 0);
    assert(info. stop < this->size[0]);
    Tensor<T> t(index.size() * this->numel(1));

    return std::move(t);
}

template<typename T>
Tensor<T> Tensor<T>::operator[](idx_type index) {
    assert(index < this->size[0]);
    SharedPtr<T> p_out(data, index * numel(1));
    Size s_out(size, 1);
    Tensor<T> out(std::move(p_out), std::move(s_out));
    return out;
}

template<typename T>
inline dim_type Tensor<T>::dims() const {
    return this->size.dims();
}

// FIXME maybe. Deprecated for now
template<typename T>
Tensor<T>::Tensor(SharedPtr<T>&& new_data, Size&& new_shape)
    : data{new_data}, size{new_shape} {
}

template<typename T>
inline  len_type Tensor<T>::getOffset(std::initializer_list<idx_type> args) const {
    //NOTE: don't really need dim checking(for now it's called only from [{...}] operator)
    if (args.size() == 1){
        return numel(1) * (*args.begin());
    }
    len_type result = 0;
    idx_type i = 1;//args.size();
    for (auto item = args.begin(); item < args.end() - 1; item++){
        result += (*item) * numel(i);
        i++;
    }
    result += *(args.end() - 1);
    return result;
}

template<typename T>
inline Tensor<T>::Tensor(std::initializer_list<len_type> shape, T fill_value) {
	this->size    = shape;
	len_type prod = this->numel(0);
	this->data.reserve(prod);
	for (len_type i = 0; i < prod; i++, fill_value++) data[i] = fill_value;
}

template<typename T>
inline Tensor<T>::Tensor(const Tensor& other) {
	this->data.reserve(other.numel());
    for (len_type i = 0; i < other.numel(); ++i)
        data[i] = other.data[i];
	this->size = other.size;
}

template<typename T>
inline Tensor<T>::Tensor(Tensor&& other) noexcept
    : data{std::move(other.data)}, size{std::move(other.size)} {
}

// for now useless, to be implemented
template<typename T >
inline Tensor<T>& Tensor<T>::operator=(const Tensor& other)
{
	return *this;
}

// for now useless, to be implemented
template<typename T>
inline Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
	return *this;
}

// TOP PRIORITY for now
template<typename T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<idx_type> indexes)  {
	assert(this->dims() != 0 && "can't subscript scalar tensor");
	assert(indexes.size() <= this->dims() && "too many indexes to subscript");
	//FIXME, horrible match checking
    idx_type arg_size = 0;
    for (len_type item : indexes) {
		assert(item < this->size[arg_size] && "index out of bounds");
	    arg_size++;
    }
    SharedPtr<T> p_out((this->data), this->getOffset(indexes));
    Size s_out(this->size, arg_size);
	return Tensor<T>(std::move(p_out), std::move(s_out));
}

template<typename T>
inline Size Tensor<T>::shape() const {
	return this->size;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& object) {
    // outputs tensor as 1d array FIXME
    out << '[';
    for (len_type i = 0; i < object.numel(0); i++){
        out << object.data[i] << ", ";
    }
    out << ']' << std::endl;
    return out;
}

template<typename T>
inline len_type Tensor<T>::numel(dim_type start_dim) const {
	assert(start_dim < this->dims() && "dim number is too high");
	len_type prod = 1;
	for (len_type i = start_dim; i < this->dims(); ++i) prod *= this->size[i];
	return prod;
}

