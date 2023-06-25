#include "utils.hpp"


Size::Size(std::initializer_list<len_type> args) {
    this->length = args.size();
    this->data = new len_type[length];
    dim_type i = 0;
    for (len_type arg: args) {
        this->data[i] = arg;
        i++;
    }
}

len_type Size::numel(dim_type start_dim) const {
    assert(start_dim < this->length);
    len_type out = 1;
    for (dim_type i = start_dim; i < this->length; i++) out *= this->data[i];
    return out;
}

dim_type Size::index(len_type find) const {
    for (dim_type i = 0; i < this->length; i++){
        if (this->data[i] == find)
            return i;
    }
    return 0;
}

dim_type Size::count(len_type find) const {
    dim_type out = 0;
    for (dim_type i = 0; i < this->length; i++){
        if (this->data[i] == find)
            out ++;
    }
    return out;
}

Size::~Size() {
    delete[] this->data;
}

Size::Size(const Size & other, dim_type start_dim) {
    assert(start_dim < other.dims());
    this->length = other.dims() - start_dim;
    this->data = new len_type[length];
    for (dim_type i = 0; i < length; i++)
        this->data[i] = other[i];
}

Size::Size(const Size & other) {
    this->length = other.dims();
    this->data = new len_type[length];
}

Size::Size(Size && other) noexcept {
    this->length = other.dims();
    this->data = other.data;
    other.data = nullptr;
}

Size &Size::operator=(const Size & other) {
    if (this == &other)
        return *this;

    if (this->data != nullptr)
        delete[] this->data;

    this->length = other.dims();
    this->data = new len_type[length];
    for (dim_type i = 0; i < length; i++)
        this->data[i] = other[i];
    return *this;
}

Size &Size::operator=(Size && other) noexcept {
    if (this == &other)
        return *this;
    if (this->data != nullptr)
        delete[] data;
    this->length = other.dims();
    this->data = other.data;
    other.data = nullptr;
    return *this;
}

Size& Size::operator=(std::initializer_list<len_type> args) {
    if (this->data != nullptr)
        delete[] data;
    this->length = args.size();
    this->data = new len_type[length];
    dim_type i = 0;
    for (len_type arg: args) {
        this->data[i] = arg;
        i ++;
    }
    return *this;
}

bool Size::operator==(std::initializer_list<len_type> args) const {
    if (this->length != args.size())
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
    if (this->length != other.dims())
        return false;
    for (dim_type i = 0; i < length; i++) {
        if (this->data[i] != other[i])
            return false;
    }
    return true;
}

bool Size::operator<(const Size & other) const {
    for (dim_type i = 0; i < MIN(this->length, other.length); i++){
        if (this->data[i] > other[i])
            return false;
    }
    return true;
}

bool Size::operator>(const Size & other) const {
    for (dim_type i = 0; i < MIN(this->length, other.length); i++){
        if (this->data[i] < other[i])
            return false;
    }
    return true;
}

bool Size::operator<(std::initializer_list<len_type> args) const {
    assert(args.size() < this->length);
    dim_type i = 0;
    for (len_type arg : args){
        if (this->data[i] > arg)
            return false;
        i ++;
    }
    return true;
}

bool Size::operator>(std::initializer_list<len_type> args) const {
    assert(args.size() < this->length);
    dim_type i = 0;
    for (len_type arg : args){
        if (this->data[i] < arg)
            return false;
        i ++;
    }
    return true;
}

len_type Size::operator[](dim_type index) const {
    assert(index < this->length);
    return this->data[index];
}

dim_type Size::dims() const {
    return this->length;
}


std::ostream& operator<<(std::ostream& out, const Size& size){
    out << "Size(";
    for (dim_type i = 0; i < size.dims(); i++)
        out << size[i] << ", ";
    out << ')' << std::endl;
    return out;
}

// -------------------------------------------------------------------



