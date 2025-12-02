#ifndef SIMPLE_JSON_H
#define SIMPLE_JSON_H

#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleJson {

enum class Type {
    Null,
    Bool,
    Number,
    String,
    Object,
    Array
};

struct Value {
    Value(Type t = Type::Null);

    Type type;
    double number_value;
    bool bool_value;
    std::string string_value;
    std::unordered_map<std::string, Value> object_values;
    std::vector<Value> array_values;

    bool isNull() const { return type == Type::Null; }
    bool isBool() const { return type == Type::Bool; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isObject() const { return type == Type::Object; }
    bool isArray() const { return type == Type::Array; }

    const Value* find(const std::string& key) const;
    Value* find(const std::string& key);
};

bool parse(const std::string& input, Value& out, std::string& error);

}  // namespace SimpleJson

#endif  // SIMPLE_JSON_H

