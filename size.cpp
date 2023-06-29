#include "size.hpp"

Size::Size(std::initializer_list<len_type> args) {
    this->ndim = args.size();
    this->data = new len_type[ndim];
    dim_type i = 0;
    for (len_type arg: args) {
        this->data[i] = arg;
        i++;
    }
}

inline len_type Size::numel(dim_type start_dim) const {
    assert(start_dim < this->ndim);
    len_type out = 1;
    for (dim_type i = start_dim; i < this->ndim; i++) out *= this->data[i];
    return out;
}

dim_type Size::index(len_type find) const {
    for (dim_type i = 0; i < this->ndim; i++){
        if (this->data[i] == find)
            return i;
    }
    return 0;
}

dim_type Size::count(len_type find) const {
    dim_type out = 0;
    for (dim_type i = 0; i < this->ndim; i++){
        if (this->data[i] == find)
            out ++;
    }
    return out;
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
    for (dim_type i = 0; i < ndim; i++)
        this->data[i] = other.data[i + start_dim];
}

Size::Size(const Size & other) {
    this->ndim = other.ndim;
    this->data   = new len_type[ndim];
    for (int i = 0; i < ndim; i++) data[i] = other.data[i];
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

    if (this->data)
        delete[] this->data;

    this->ndim = other.ndim;
    this->data = new len_type[ndim];
    for (dim_type i = 0; i < ndim; i++)
        this->data[i] = other[i];
    return *this;
}

Size &Size::operator=(Size && other) noexcept {
    if (this == &other)
        return *this;

    if (this->data)
        delete[] this->data;

    this->ndim = other.ndim;
    this->data   = other.data;

    other.data = nullptr;
    other.ndim = 0;   // optional

    return *this;
}

Size& Size::operator=(std::initializer_list<len_type> args) {
    if (this->data)
        delete[] this->data;

    this->ndim = args.size();
    this->data   = new len_type[ndim];

    dim_type i = 0;
    for (len_type arg: args) {
        this->data[i] = arg;
        i ++;
    }

    return *this;
}

bool Size::operator==(std::initializer_list<len_type> args) const {
    if (this->ndim != args.size())
        return false;
    dim_type i = 0;
    for (len_type arg: args) {
        if (this->data[i] != arg)
            return false;
        i ++;
    }
    return true;
}

bool Size::operator==(const Size & other) const {
    if (this->ndim != other.dims())
        return false;
    for (dim_type i = 0; i < ndim; i++) {
        if (this->data[i] != other[i])
            return false;
    }
    return true;
}

bool Size::operator<(const Size & other) const {
    for (dim_type i = 0; i < MIN(this->ndim, other.ndim); i++){
        if (this->data[i] > other[i])
            return false;
    }
    return true;
}

bool Size::operator>(const Size & other) const {
    for (dim_type i = 0; i < MIN(this->ndim, other.ndim); i++){
        if (this->data[i] < other[i])
            return false;
    }
    return true;
}

bool Size::operator<(std::initializer_list<len_type> args) const {
    assert(args.size() < this->ndim);
    dim_type i = 0;
    for (len_type arg : args){
        if (this->data[i] > arg)
            return false;
        i ++;
    }
    return true;
}

bool Size::operator>(std::initializer_list<len_type> args) const {
    assert(args.size() < this->ndim);
    dim_type i = 0;
    for (len_type arg : args){
        if (this->data[i] < arg)
            return false;
        i ++;
    }
    return true;
}

inline len_type Size::operator[](dim_type index) const {
    assert(index < this->ndim);
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
