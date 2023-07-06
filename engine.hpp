#pragma once

#include "size.hpp"
#include "smartptr.hpp"
#include <memory>
#include <cassert>
#include <initializer_list>
#include <ostream>
#include <array>
#include <functional>
#include "deprecated.hpp"   // shall be deleted soon

template<Algebraic T>
class Tensor {
public:
	Tensor() = default;
	Tensor(std::initializer_list<len_type>, T);
	Tensor(const Tensor<T>&, bool empty = false);
	Tensor(Tensor<T>&&) noexcept;
    explicit Tensor(const Size&);
    explicit Tensor(len_type);

    Comparison compare (const Tensor<T>&);
    T& item() const;
	[[nodiscard]] Size     shape() const;
    [[nodiscard]] dim_type dims()  const;
    [[nodiscard]] len_type numel(dim_type start_dim = 0) const;
    [[nodiscard]] len_type getOffset(std::initializer_list<idx_type>) const;

    Tensor<T> empty_like();
	Tensor<T>& operator = (Tensor&);
	Tensor<T>& operator = (Tensor&&) noexcept;

    // NOTE: all these ops share pointer to data. Check for it.
    Tensor<T> operator [] (idx_type);       // Ready
	Tensor<T> operator [] (std::initializer_list<len_type>);    // no multiple arguments until c++23
    Tensor<T> operator [] (const Tensor<bool>&) = delete;       // THINK: maybe useless(move version is more popular)
    Tensor<T> operator [] (Tensor<bool>&&)      = delete;       // will be implemented after boolean ops
    [[deprecated("useless for now")]] Tensor<T> operator [] (Range<len_type>&&);// indexing array without creating new Range instance

    Tensor<bool> operator == (const Tensor<T>&) const = delete;
    Tensor<bool> operator == (T)                const = delete;
    Tensor<bool> operator > (const Tensor<T>&)  const = delete;
    Tensor<bool> operator > (T)                 const = delete;
    Tensor<bool> operator < (const Tensor<T>&)  const = delete;
    Tensor<bool> operator < (T)                 const = delete;

    Tensor<T>  bin_op(Tensor<T>&, std::function<void(T&, T)>);
    Tensor<T>& bin_op_ip(Tensor<T>&, std::function<void(T&, T)>);
    Tensor<T>  un_op(std::function<void(T&)>);
    Tensor<T>& un_op_ip(std::function<void(T&)>);

    Tensor<T> operator + (Tensor<T>&);        // continue from here
    Tensor<T> operator + (T)                 = delete;
    Tensor<T> operator - (const Tensor<T>&)  = delete;
    Tensor<T> operator - (T)                 = delete;
    Tensor<T> operator * (const Tensor<T>&)  = delete;
    Tensor<T> operator * (T)                 = delete;
    Tensor<T> operator / (const Tensor<T>&)  = delete;
    Tensor<T> operator / (T)                 = delete;
    Tensor<T> operator ^ (double)            = delete;  // analogue of __pow__
    Tensor<T> operator ^ (int)               = delete;  // analogue of __pow__

    Tensor<T> operator - ();

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


    friend std::ostream& operator<<(std::ostream& out, const Tensor<T>& object) {
        //FIXME outputs tensor as 1d array
        // should probably be recursive
        out << '[';
        for (len_type i = 0; i < object.numel(0); i++){
            out << object.data[i] << ", ";
        }
        out << ']' << std::endl;
        return out;
    }
private:
	SharedPtr<T> data;  // FIXME check for compatibility
    Size size;
    // place for a lambda _backward function: std::function<void(...)> _backward = nullptr;
    // place for another SharedPtr<T> holding gradient values: SharedPtr<T> grad;
    // place for a boolean grad flag: bool requires_grad = false;
    // place for (some version of array) of parent Tensors.
	Tensor(SharedPtr<T>&&, Size&&);
};

template<Algebraic T>
T& Tensor<T>::item() const {
    assert(numel() == 1 && size.dims() == 1);
    return data[0];
}

template<Algebraic T>
Tensor<T>& Tensor<T>::bin_op_ip(Tensor<T>& other, std::function<void(T&, T)> operation) {
    if (other.size == size) {    // basic case
        util::apply_ip<T>(data.data(), numel(), other.data.data(), operation);
        return *this;
    }
    Comparison state = this->compare(other);
    T* out; len_type out_s;
    T* min_t; len_type min_s;
    switch (state) {
        case Comparison::gt: {
            out   = this->data.data(); out_s = this->numel();
            min_t = other.data.data(); min_s = other.numel();
            break;
        }
        case Comparison::lt: {
            out   = other.data.data(); out_s = other.numel();
            min_t = this->data.data(); min_s = this->numel();
            break;
        }
        case Comparison::ne: {
            assert(0 && "Shapes of tensors don't match");
            break;
        }
        default: {
            assert(0 && "Can't happen");
        }
    }
    for (int block = 0; block < out_s; block += min_s)
        for (int i = 0; i < min_s; i++)
            operation(out[block + i], min_t[i]);   // FIXME: should be faster if replaced with inplace function
    return *this;
}

template<Algebraic T>
Comparison Tensor<T>::compare(const Tensor<T> & other) {
    return this->size.compare(other.size);
}

template<Algebraic T>
Tensor<T>::Tensor(const Size & new_size) {
    this->size = new_size;
    this->data.reserve(this->size.numel());
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator-() {
    return this->un_op([](T x) -> T{ return -x; });
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T> & other) {
    this->bin_op_ip(other, [](T& mut, T arg) -> void{ mut += arg; });
    return *this;
}

template<Algebraic T>
Tensor<T> Tensor<T>::un_op(std::function<void(T&)> operation) {
    Tensor<T> out(size);
    std::transform(data.data(), data.data() + numel(), out.data.data(), operation);
    return std::move(out);
}

// FIXME: implement a lot of helper - function and optimize the process, which is bloated for now
template<Algebraic T>
Tensor<T> Tensor<T>::bin_op(Tensor<T> & other, std::function<void(T&, T)> operation) {
    Tensor<T> out;
    if (numel() < other.numel()) {
        out = this;
        out.bin_op_ip(other, operation);
    }
    else {
        out = other;
        out.bin_op_ip(*this, operation);
    }
    return out;
}


template<Algebraic T>
Tensor<T> Tensor<T>::operator+(Tensor<T> & other) {
    return this->bin_op(other, [](T x, T y) -> T { return x + y; });
}

template<Algebraic T>
Tensor<T>::Tensor(len_type length) {
    this->data.reserve(length);
}

// FIXME: implement me
template<Algebraic T>
[[deprecated("too complicated for now")]] Tensor<T> Tensor<T>::operator[](Range<len_type> && index) {
    return *this;
    auto info = index.get_info();
    assert(info.start >= 0);
    assert(info.stop < this->size[0]);
    Tensor<T> t(index.size() * this->numel(1));

    return std::move(t);
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator[](idx_type index) {
    assert(index < this->size[0]);
    SharedPtr<T> p_out(data, index * numel(1));
    Size s_out(size, 1);
    Tensor<T> out(std::move(p_out), std::move(s_out));
    return out;
}

template<Algebraic T>
inline dim_type Tensor<T>::dims() const {
    return this->size.dims();
}

// FIXME maybe. Deprecated for now
template<Algebraic T>
Tensor<T>::Tensor(SharedPtr<T>&& new_data, Size&& new_shape)
    : data{new_data}, size{new_shape} {
}

template<Algebraic T>
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

template<Algebraic T>
inline Tensor<T>::Tensor(std::initializer_list<len_type> shape, T fill_value) {
	this->size    = shape;
	len_type prod = this->numel(0);
	this->data.reserve(prod);
	for (len_type i = 0; i < prod; i++, fill_value++) data[i] = fill_value;
}

template<Algebraic T>
inline Tensor<T>::Tensor(const Tensor& other, bool empty) {
	this->data.reserve(other.numel());
    if (!empty)
        for (len_type i = 0; i < other.numel(); ++i)
            data[i] = other.data[i];
	this->size = other.size;
}

template<Algebraic T>
inline Tensor<T>::Tensor(Tensor&& other) noexcept
    : data{std::move(other.data)}, size{std::move(other.size)} {
}

// for now useless, to be implemented
template<Algebraic T >
inline Tensor<T>& Tensor<T>::operator=(Tensor& other) {
    if (this == &other)
        return *this;
    this->data = other.data;
    this->size = other.size;
	return *this;
}

// for now useless, to be implemented
template<Algebraic T>
inline Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    this->data = std::move(other.data);
    this->size = std::move(other.size);
	return *this;
}

// TOP PRIORITY for now
template<Algebraic T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<len_type> indexes)  {
	assert(this->dims() != 0 && "can't subscript scalar tensor");
	assert(indexes.size() <= this->dims() && "too many indexes to subscript");
	//FIXME, horrible match checking
    len_type arg_size = indexes.size();
    assert(size > indexes && "Not compatible sizes");
    SharedPtr<T> p_out((this->data), this->getOffset(indexes));
    Size s_out(this->size, arg_size);
	return Tensor<T>(std::move(p_out), std::move(s_out));
}

template<Algebraic T>
inline Size Tensor<T>::shape() const {
	return this->size;
}



template<Algebraic T>
inline len_type Tensor<T>::numel(dim_type start_dim) const {
	assert(start_dim < this->dims() && "dim number is too high (numel)");
	len_type prod = 1;
	for (len_type i = start_dim; i < this->dims(); ++i) prod *= this->size[i];
	return prod;
}

