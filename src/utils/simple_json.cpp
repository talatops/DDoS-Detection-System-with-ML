#include "utils/simple_json.h"
#include <cctype>
#include <stdexcept>

namespace SimpleJson {

Value::Value(Type t) : type(t), number_value(0.0), bool_value(false) {}

const Value* Value::find(const std::string& key) const {
    if (!isObject()) return nullptr;
    auto it = object_values.find(key);
    if (it == object_values.end()) return nullptr;
    return &it->second;
}

Value* Value::find(const std::string& key) {
    if (!isObject()) return nullptr;
    auto it = object_values.find(key);
    if (it == object_values.end()) return nullptr;
    return &it->second;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& input) : input_(input), pos_(0) {}

    bool parse(Value& out, std::string& error) {
        skipWhitespace();
        if (!parseValue(out)) {
            error = error_;
            return false;
        }
        skipWhitespace();
        if (pos_ != input_.size()) {
            error = "Trailing characters after JSON content";
            return false;
        }
        return true;
    }

private:
    const std::string& input_;
    size_t pos_;
    std::string error_;

    bool parseValue(Value& out) {
        if (pos_ >= input_.size()) {
            error_ = "Unexpected end of input";
            return false;
        }
        char c = input_[pos_];
        if (c == '{') {
            return parseObject(out);
        } else if (c == '[') {
            return parseArray(out);
        } else if (c == '"') {
            out = Value(Type::String);
            return parseString(out.string_value);
        } else if (c == 't') {
            return parseLiteral("true", Value(Type::Bool), out, true);
        } else if (c == 'f') {
            return parseLiteral("false", Value(Type::Bool), out, false);
        } else if (c == 'n') {
            return parseLiteral("null", Value(Type::Null), out, false);
        } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parseNumber(out);
        }
        error_ = "Invalid character in JSON";
        return false;
    }

    bool parseObject(Value& out) {
        out = Value(Type::Object);
        ++pos_;  // skip '{'
        skipWhitespace();
        if (pos_ < input_.size() && input_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (pos_ < input_.size()) {
            Value key_val(Type::String);
            if (!parseString(key_val.string_value)) {
                error_ = "Expected string key in object";
                return false;
            }
            skipWhitespace();
            if (!consume(':')) {
                error_ = "Expected ':' after key";
                return false;
            }
            skipWhitespace();
            Value value;
            if (!parseValue(value)) {
                return false;
            }
            out.object_values.emplace(key_val.string_value, std::move(value));
            skipWhitespace();
            if (consume('}')) {
                return true;
            }
            if (!consume(',')) {
                error_ = "Expected ',' between object members";
                return false;
            }
            skipWhitespace();
        }
        error_ = "Unterminated object";
        return false;
    }

    bool parseArray(Value& out) {
        out = Value(Type::Array);
        ++pos_;  // skip '['
        skipWhitespace();
        if (pos_ < input_.size() && input_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (pos_ < input_.size()) {
            Value element;
            if (!parseValue(element)) {
                return false;
            }
            out.array_values.emplace_back(std::move(element));
            skipWhitespace();
            if (consume(']')) {
                return true;
            }
            if (!consume(',')) {
                error_ = "Expected ',' between array elements";
                return false;
            }
            skipWhitespace();
        }
        error_ = "Unterminated array";
        return false;
    }

    bool parseString(std::string& out) {
        if (input_[pos_] != '"') {
            error_ = "Expected string opening quote";
            return false;
        }
        ++pos_;
        while (pos_ < input_.size()) {
            char c = input_[pos_++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (pos_ >= input_.size()) {
                    error_ = "Incomplete escape sequence";
                    return false;
                }
                char esc = input_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        error_ = "Unsupported escape sequence";
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        error_ = "Unterminated string";
        return false;
    }

    bool parseNumber(Value& out) {
        size_t start = pos_;
        if (input_[pos_] == '-') ++pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        try {
            double value = std::stod(input_.substr(start, pos_ - start));
            out = Value(Type::Number);
            out.number_value = value;
            return true;
        } catch (const std::exception&) {
            error_ = "Invalid number format";
            return false;
        }
    }

    bool parseLiteral(const std::string& literal, Value proto, Value& out, bool bool_value) {
        if (input_.compare(pos_, literal.size(), literal) == 0) {
            pos_ += literal.size();
            out = proto;
            if (proto.type == Type::Bool) {
                out.bool_value = bool_value;
            }
            return true;
        }
        error_ = "Invalid literal";
        return false;
    }

    bool consume(char expected) {
        if (pos_ < input_.size() && input_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }
};

}  // namespace

bool parse(const std::string& input, Value& out, std::string& error) {
    Parser parser(input);
    return parser.parse(out, error);
}

}  // namespace SimpleJson

