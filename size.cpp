#include "size.hpp"

Size::Size(std::initializer_list<len_type> args) {
    this->data_ = args;
}

len_type Size::numel(dim_type start_dim, int end_dim) const {
    assert(start_dim < this->ndim());
    auto end_index = util::index_abs(end_dim, ndim());
    assert(end_index <= this->ndim());
    assert(start_dim < end_index);
    if (!ndim())
        return 0;
    len_type out = 1;
    std::for_each(data_.begin() + start_dim, data_.begin() + end_index,
                  [&out](len_type x)->void{ out *= x; });
    return out;
}

dim_type Size::index(len_type find) const {
    auto ret = std::find(data_.begin(), data_.end(), find);
    return ret - data_.begin();
}

dim_type Size::count(len_type find) const {
    return std::count(this->data_.begin(), this->data_.end(), find);
}

//FIXME im ugly as fuck. Can be unfixable
Size::Size(const Size & other, dim_type start_dim) {
    if (other.ndim() == start_dim){
        this->data_ = {1, };
        return;
    }
    assert(start_dim <= other.ndim());
    this->data_.resize(other.ndim() - start_dim);
    std::copy(other.data_.begin() + start_dim, other.data_.end(),this->data_.begin());
    std::cout << *this << int(start_dim) << std::endl;  // FIXME COUT
}

Size::Size(const Size & other) {
    this->data_ = other.data_;
}

Size::Size(Size && other) noexcept {
    this->data_ = std::move(other.data_);
}

Size& Size::operator=(Size && other) noexcept {
    this->data_ = std::move(other.data_);
    return *this;
}

Size& Size::operator=(std::initializer_list<len_type> args) {
    this->data_ = args;
    return *this;
}

bool Size::operator==(std::initializer_list<len_type> args) const {
    if (this->ndim() != args.size())
        return false;
    return std::equal(args.begin(), args.end(), this->data_.begin());
}

bool Size::operator==(const Size & other) const {
    if (this->ndim() != other.dims())
        return false;
    return std::equal(this->data_.begin(), this->data_.end(),
                      other.data_.begin());
}

bool Size::operator<(const Size & other) const {
    if (this->ndim() != other.ndim())
        return false;
    return std::equal(this->data_.begin(), this->data_.end(),
                      other.data_.begin(),[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(const Size & other) const {
    if (this->ndim() != other.ndim())
        return false;
    return std::equal(this->data_.begin(), this->data_.end(),
                      other.data_.begin(),[](len_type t, len_type o)->bool{ return t > o; });
}

bool Size::operator<(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim())
        return false;
    return std::equal(this->data_.begin(), this->data_.end(),
                      args.begin(),[](len_type t, len_type o)->bool{ return t < o; });
}

bool Size::operator>(std::initializer_list<len_type> args) const {
    if (args.size() != this->ndim())
        return false;
    return std::equal(this->data_.begin(), this->data_.end(),
                      args.begin(),[](len_type t, len_type o)->bool{ return t > o; });
}

len_type& Size::operator[](int index) const {
    assert((index < ndim() || index > -uint8_t(ndim())) && "out of bounds");
    index = (index >= 0)?index:(ndim() + index);
    return const_cast<len_type&> (this->data_[index]);
}

inline dim_type Size::dims() const {
    return this->data_.size();
}

std::ostream& operator<<(std::ostream& out, const Size& size){
    out << "Size(";
    for (auto i : size.data_)
        out << i << ", ";
    out << ')' << std::endl;
    return out;
}

// FIXME: too much branches
// TODO: AM I A КОСТЫЛЬ?
Comparison Size::compare(const Size & other, int skip_from_end) const {
    if (skip_from_end == MIN(ndim(), other.ndim()))
        return Comparison::eq;
    assert(skip_from_end < MIN(ndim(), other.ndim()));
    if (*this == other)
        return Comparison::eq;
    Comparison cmp = Comparison::ne;
    for (auto i = ndim() - 1 - skip_from_end, j = other.ndim() - 1 - skip_from_end; i >= 0 && j >= 0; --i, --j) {
        if (data_[i] == other.data_[j] || data_[i] == 1 || data_[j] == 1) {
            cmp = Comparison::eq;
        } else {
            return cmp;
        }
    }
    if (cmp == Comparison::eq) {    // NOTE: this condition might be useless
        if (ndim() > other.ndim())
            return Comparison::gt;
        else if (ndim() < other.ndim())
            return Comparison::lt;
    }
    return cmp;
}

auto Size::begin() const -> decltype(data_.begin()) {
    return data_.begin();
}

auto Size::end() const -> decltype(data_.end()) {
    return data_.end();
}

Size::Size(dim_type ndim) {
    assert(ndim != 0);
    this->data_.reserve(ndim);
}

// NOTE: a lot of branching
Size Size::remove(dim_type skip, bool keepdim) const {
    assert(skip < ndim());
    std::vector<len_type> out = data_;
    if (keepdim)
        out[skip] = 1;
    else
        out.erase(out.begin() + skip);
    return {out};
}

len_type Size::size() const {
    return data_.size();
}

Size Size::insert(dim_type dim, len_type value) const {
    auto out = data_;
    out.insert(out.begin() + dim, value);
    return {out};
}

Size::Size(const std::vector<len_type>& other) {
    data_ = other;
}

dim_type Size::ndim() const {
    return data_.size();
}

bool Size::is_scalar() const {
    if (size() == 1 && data_[0] == 1)
        return true;
    return false;
}

Size Size::slice(int start, int end, int step) const {
    start = (start >= 0)?start:(ndim() + start);
    end = (end >= 0)?end:(ndim() + end);
    std::vector<len_type> out(size());
    auto index_abs = [this](int idx) -> int { return (idx >= 0)?idx:(ndim() + idx); };
    for (int i = start; index_abs(i) < end; i += step) {
        out.push_back((*this)[i]);
    }
    return {out};
}

Size::Size(std::vector<len_type> && other): data_{std::move(other)} {}

auto Size::data() -> decltype(data_.data()) const {
    return data_.data();
}

