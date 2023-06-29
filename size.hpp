#pragma once

#include "utils.hpp"

class Size{
private:
    len_type*  data = nullptr;
    dim_type ndim = 0;
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
    dim_type dims() const;
    len_type numel(dim_type start_dim = 0) const;
    dim_type index(len_type) const;
    dim_type count(len_type) const;

    ~Size();

    friend std::ostream& operator <<(std::ostream&, const Size&);
};
