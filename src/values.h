/*
 * Copyright (C) 2015 Emmanuel Durand
 *
 * This file is part of GifBox.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GifBox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GifBox.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VALUES_H
#define VALUES_H

#include <string>
#include <vector>

struct Value;
using Values = std::vector<Value>;

/*************/
struct Value
{
    public:
        enum Type
        {
            i = 0,
            f,
            s,
            v
        };

        Value() {_i = 0; _type = Type::i;}
        Value(int v) {_i = v; _type = Type::i;}
        Value(float v) {_f = v; _type = Type::f;}
        Value(double v) {_f = (float)v; _type = Type::f;}
        Value(std::string v) {_s = v; _type = Type::s;}
        Value(const char* c) {_s = std::string(c); _type = Type::s;}
        Value(Values v) {_v = v; _type = Type::v;}

        template<class InputIt>
        Value(InputIt first, InputIt last)
        {
            _type = Type::v;
            _v.clear();

            auto it = first;
            while (it != last)
            {
                _v.push_back(Value(*it));
                ++it;
            }
        }

        bool operator==(Value v) const
        {
            if (_type != v._type)
                return false;
            else if (_type == Type::i)
                return _i == v._i;
            else if (_type == Type::f)
                return _f == v._f;
            else if (_type == Type::s)
                return _s == v._s;
            else if (_type == Type::v)
            {
                if (_v.size() != v._v.size())
                    return false;
                bool isEqual = true;
                for (unsigned int i = 0; i < _v.size(); ++i)
                    isEqual &= (_v[i] == v._v[i]);
                return isEqual;
            }
            else
                return false;
        }

        int asInt() const
        {
            if (_type == Type::i)
                return _i;
            else if (_type == Type::f)
                return (int)_f;
            else if (_type == Type::s)
                try {return std::stoi(_s);}
                catch (...) {return 0;}
            else
                return 0;
        }

        float asFloat() const
        {
            if (_type == Type::i)
                return (float)_i;
            else if (_type == Type::f)
                return _f;
            else if (_type == Type::s)
                try {return std::stof(_s);}
                catch (...) {return 0.f;}
            else
                return 0.f;
        }

        std::string asString() const
        {
            if (_type == Type::i)
                try {return std::to_string(_i);}
                catch (...) {return std::string();}
            else if (_type == Type::f)
                try {return std::to_string(_f);}
                catch (...) {return std::string();}
            else if (_type == Type::s)
                return _s;
            else
                return "";
        }

        Values asValues() const
        {
            if (_type == Type::i)
                return {_i};
            else if (_type == Type::f)
                return {_f};
            else if (_type == Type::s)
                return {_s};
            else if (_type == Type::v)
                return _v;
            else
                return {};
        }

        void* data()
        {
            if (_type == Type::i)
                return (void*)&_i;
            else if (_type == Type::f)
                return (void*)&_f;
            else if (_type == Type::s)
                return (void*)_s.c_str();
            else
                return nullptr;
        }

        Type getType() const {return _type;}
        
        int size()
        {
            if (_type == Type::i)
                return sizeof(_i);
            else if (_type == Type::f)
                return sizeof(_f);
            else if (_type == Type::s)
                return _s.size();
            else
                return 0;
        }

    private:
        Type _type;
        int _i {0};
        float _f {0.f};
        std::string _s {""};
        Values _v {};
};

#endif
