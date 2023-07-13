#include "size.hpp"

Size::Size(std::initializer_list<len_type> args) {
    this->ndim = args.size();
    this->data_ = new len_type[ndim];
    dim_type i = 0;
    std::copy(args.begin(), args.end(), this->data_);
}

len_type Size::numel(dim_type start_dim) const {
    assert(start_dim < this->ndim);
    if (!ndim)
        return 0;
    len_type out = 1;
    std::for_each(data_, data_ + ndim, [&out](len_type x)->void{out *= x;});
    return out;
}

dim_type Size::index(len_type find) const {
    auto ret = std::find(data_, data_ + ndim, find);
    return (ret == data_ + ndim)?0:*ret;
}

dim_type Size::count(len_type find) const {
    return std::count(this->data_, this->data_ + ndim, find);
}

Size::~Size() {
    delete[] this->data_;
}

//FIXME im ugly as fuck. Can be unfixable
Size::Size(const Size & other, dim_type start_dim) {
    if (this->ndim == start_dim){
        this->ndim = 1;
        this->data_ = new len_type;
        *data_ = 1;
        return;
    }
    assert(start_dim <= other.ndim);
    this->ndim = other.ndim - start_dim;
    this->data_ = new len_type[ndim];
    std::copy(other.data_ + start_dim, other.data_ + start_dim + ndim,this->data_);
}

Size::Size(const Size & other) {
    this->ndim = other.ndim;
    this->data_   = new len_type[ndim];
    std::copy(other.data_, other.data_ + ndim, this->data_);
}

Size::Size(Size && other) noexcept {
    this->ndim = other.ndim;
    this->data_ = other.data_;

    other.data_ = nullptr;
    other.ndim = 0;    // optional
}

Size& Size::operator=(const Size & other) {
    if (this == &other)
        return *this;

    if (this->data_) {
        if (this->ndim == other.ndim)
            goto copy_data_;
        delete[] this->data_;
    }
    this->ndim = other.ndim;
    this->data_ = new len_type[ndim];
    copy_data_:
    std::copy(other.data_, other.data_ + ndim, this->data_);
    return *this;
}

Size& Size::operator=(Size && other) noexcept {
    if (this->data_)
        delete[] this->data_;

    this->ndim   = other.ndim;
    this->data_   = other.data_;

    other.data_ = nullptr;
    other.ndim = 0;   // optional

    return *this;
}

Size& Size::operator=(std::initializer_list<len_type> args) {
    if (this->data_) {
        if (this->ndim == args.size())
            goto copy_data_;
        delete[] this->data_;
    }
    this->ndim = args.size();
    this->data_ = new len_type[ndim];
    copy_data_:
    std::copy(args.begin(), args.end(), this->data_);
    return *this;
}

bool Size::operator==(std::initializer_list<len_type> args) const {
    if (this->ndim != args.size())
        return false;
    return std::equal(args.begin(), args.end(), this->data_);
}

bool Size::operator==(const Size & other) const {
    if (this->ndim != other.dims())
        return false;
    return std::equal(this->data_, this->data_ + ndim,
                      other.data_);
}

bool Size::operator<(const Size & other) const {
    if (this->ndim != other.ndim)
        return false;
    return std::equal(this->data_, this->data_ + ndim,
                      other.data_,[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(const Size & other) const {
    if (this->ndim != other.ndim)
        return false;
    return std::equal(this->data_, this->data_ + ndim,
                      other.data_,[](len_type t, len_type o)->bool{ return t > o; });
}

bool Size::operator<(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim)
        return false;
    return std::equal(this->data_, this->data_ + ndim,
                      args.begin(),[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim)
        return false;
    return std::equal(this->data_, this->data_ + ndim,
                      args.begin(),[](len_type t, len_type o)->bool{ return t > o; });
}

inline len_type& Size::operator[](dim_type index) const {
    assert(index < this->ndim && "out of bounds");
    return this->data_[index];
}

inline dim_type Size::dims() const {
    return this->ndim;
}

std::ostream& operator<<(std::ostream& out, const Size& size){
    out << "Size(";
    for (dim_type i = 0; i < size.dims(); i++)
        out << size[i] << ", ";
    out << ')' << std::endl;
    return out;
}

// too much branches
Comparison Size::compare(const Size & other) const {
    if (*this == other)
        return Comparison::eq;
    Comparison cmp = Comparison::ne;
    for (auto i = ndim - 1, j = other.ndim - 1; i >= 0 && j >= 0; --i, --j) {
        if (data_[i] != other.data_[j]) {
            return cmp;
        }
        cmp = Comparison::eq;   // NOTE: can be optimized
    }
    if (cmp == Comparison::eq) {
        if (ndim > other.ndim)
            return Comparison::gt;
        return Comparison::lt;
    }
    return Comparison::ne;
}

ContiguousIterator<len_type> Size::begin() const {
    return {this->data_};
}

ContiguousIterator<len_type> Size::end() const {
    return {this->data_ + this->dims()};
}

Size::Size(dim_type ndim_) {
    assert(ndim_ != 0);
    this->ndim = ndim_;
    this->data_ = new len_type[ndim_];
}

// NOTE: a lot of branching
Size Size::remove(dim_type skip, bool keepdim) const {
    assert(skip < ndim);
    Size out(ndim - !keepdim);
    for (int i = 0, j = 0; i < ndim; ++i, ++j) {
        if (i == skip) {
            if (!keepdim)
                --j;
            else
                out.data_[j] = 1;
        } else
            out.data_[j] = data_[i];
    }
    return out;
}

len_type Size::size() const {
    return this->ndim;
}

Size Size::insert(dim_type, bool keepdim) const {
    return Size();
}

