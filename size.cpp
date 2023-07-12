#include "size.hpp"

Size::Size(std::initializer_list<len_type> args) {
    this->ndim = args.size();
    this->data = new len_type[ndim];
    dim_type i = 0;
    std::copy(args.begin(), args.end(), this->data);
}

len_type Size::numel(dim_type start_dim) const {
    assert(start_dim < this->ndim);
    if (!ndim)
        return 0;
    len_type out = 1;
    std::for_each(data, data + ndim, [&out](len_type x)->void{out *= x;});
    return out;
}

dim_type Size::index(len_type find) const {
    auto ret = std::find(data, data + ndim, find);
    return (ret == data + ndim)?0:*ret;
}

dim_type Size::count(len_type find) const {
    return std::count(this->data, this->data + ndim, find);
}

Size::~Size() {
    delete[] this->data;
}

//FIXME im ugly as fuck. Can be unfixable
Size::Size(const Size & other, dim_type start_dim) {
    if (this->ndim == start_dim){
        this->ndim = 1;
        this->data = new len_type;
        *data = 1;
        return;
    }
    assert(start_dim <= other.ndim);
    this->ndim = other.ndim - start_dim;
    this->data = new len_type[ndim];
    std::copy(other.data + start_dim, other.data + start_dim + ndim,this->data);
}

Size::Size(const Size & other) {
    this->ndim = other.ndim;
    this->data   = new len_type[ndim];
    std::copy(other.data, other.data + ndim, this->data);
}

Size::Size(Size && other) noexcept {
    this->ndim = other.ndim;
    this->data = other.data;

    other.data = nullptr;
    other.ndim = 0;    // optional
}

Size& Size::operator=(const Size & other) {
    if (this == &other)
        return *this;

    if (this->data) {
        if (this->ndim == other.ndim)
            goto copy_data;
        delete[] this->data;
    }
    this->ndim = other.ndim;
    this->data = new len_type[ndim];
    copy_data:
        std::copy(other.data, other.data + ndim, this->data);
    return *this;
}

Size &Size::operator=(Size && other) noexcept {
    if (this->data)
        delete[] this->data;

    this->ndim = other.ndim;
    this->data   = other.data;

    other.data = nullptr;
    other.ndim = 0;   // optional

    return *this;
}

Size& Size::operator=(std::initializer_list<len_type> args) {
    if (this->data) {
        if (this->ndim == args.size())
            goto copy_data;
        delete[] this->data;
    }
    this->ndim = args.size();
    this->data = new len_type[ndim];
    copy_data:
        std::copy(args.begin(), args.end(), this->data);
    return *this;
}

bool Size::operator==(std::initializer_list<len_type> args) const {
    if (this->ndim != args.size())
        return false;
    return std::equal(args.begin(), args.end(), this->data);
}

bool Size::operator==(const Size & other) const {
    if (this->ndim != other.dims())
        return false;
    return std::equal(this->data, this->data + ndim,
                      other.data);
}

bool Size::operator<(const Size & other) const {
    if (this->ndim != other.ndim)
        return false;
    return std::equal(this->data, this->data + ndim,
                      other.data,[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(const Size & other) const {
    if (this->ndim != other.ndim)
        return false;
    return std::equal(this->data, this->data + ndim,
                      other.data,[](len_type t, len_type o)->bool{ return t > o; });
}

bool Size::operator<(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim)
        return false;
    return std::equal(this->data, this->data + ndim,
                      args.begin(),[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim)
        return false;
    return std::equal(this->data, this->data + ndim,
                      args.begin(),[](len_type t, len_type o)->bool{ return t > o; });
}

inline len_type Size::operator[](dim_type index) const {
    assert(index < this->ndim && "out of bounds");
    return this->data[index];
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
Comparison Size::compare(const Size & other) {
    if (*this == other)
        return Comparison::eq;
    Comparison cmp = Comparison::ne;
    for (auto i = ndim - 1, j = other.ndim - 1; i >= 0 && j >= 0; --i, --j) {
        if (data[i] != other.data[j]) {
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

ContiguousIterator<len_type> Size::begin() {
    return {this->data};
}

ContiguousIterator<len_type> Size::end() {
    return {this->data + this->dims()};
}

Size::Size(dim_type ndim_) {
    assert(ndim_ != 0);
    this->ndim = ndim_;
    this->data = new len_type[ndim_];
}

// NOTE: a lot of branching
inline Size Size::copy_except(dim_type skip, bool keepdim) const {
    assert(skip < ndim);
    Size out(ndim - !keepdim);
    for (int i = 0, j = 0; i < ndim; ++i, ++j) {
        if (i == skip) {
            if (!keepdim)
                --j;
            else
                out.data[j] = 1;
        } else
            out.data[j] = data[i];
    }
    return out;
}
