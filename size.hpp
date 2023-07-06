#pragma once

#include <optional>
#include <tuple>
#include "utils.hpp"

#if 0
class Iter {
    len_type val;
public:
    Iter operator * () const;
    Iter& operator ++ ();
    Iter& operator ++ (int);
};
#endif

class Size {
private:
    len_type* data = nullptr;
    dim_type  ndim = 0;
public:
    Size() = default;
    Size(std::initializer_list<len_type>);
    Size(const Size&, dim_type);    // creates a copy of Size from given dim
    Size(const Size&);
    Size(Size&&) noexcept;

    Size& operator =(const Size&);
    Size& operator =(Size&&) noexcept;
    Size& operator =(std::initializer_list<len_type>);

    bool operator ==(std::initializer_list<len_type>) const;
    bool operator ==(const Size&) const;
    bool operator <(const Size&) const;
    bool operator >(const Size&) const;
    bool operator <(std::initializer_list<len_type>) const;
    bool operator >(std::initializer_list<len_type>) const;

    len_type operator[](dim_type) const;
    [[nodiscard]] Size& slice(dim_type start, dim_type end) const;  // FIXME: make compatible with std::span. RESEARCH RANGES!!!
    [[nodiscard]] Size& slice(dim_type start) const;
    [[nodiscard]] Comparison compare(const Size&);
    [[nodiscard]] dim_type dims() const;
    [[nodiscard]] len_type numel(dim_type start_dim = 0) const;
    [[nodiscard]] dim_type index(len_type) const;
    [[nodiscard]] dim_type count(len_type) const;

    ~Size();

    friend std::ostream& operator <<(std::ostream&, const Size&);
};
