#include "minijson.h"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace minijson {

static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
}

std::string escape(const std::string& s) {
    std::string out; out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static bool parse_json_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i; // skip opening quote
    out.clear();
    while (i < s.size()) {
        unsigned char c = s[i++];
        if (c == '"') return true;
        if (c == '\\') {
            if (i >= s.size()) return false;
            unsigned char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    // Skip 4 hex digits; we won't handle full unicode here.
                    for (int k = 0; k < 4 && i < s.size(); ++k) ++i;
                    // Placeholder: ignore unicode conversion; append '?'
                    out += '?';
                } break;
                default: out += (char)e; break;
            }
        } else {
            out += (char)c;
        }
    }
    return false;
}

static bool find_key_value_pos(const std::string& s, const std::string& key, size_t& val_pos) {
    // naive search for "key" followed by ':'
    const std::string pat = std::string("\"") + key + "\"";
    size_t pos = 0;
    while (true) {
        pos = s.find(pat, pos);
        if (pos == std::string::npos) return false;
        size_t colon = s.find(':', pos + pat.size());
        if (colon == std::string::npos) return false;
        val_pos = colon + 1;
        return true;
    }
}

bool extract_string(const std::string& json, const std::string& key, std::string& out) {
    size_t vpos = 0; if (!find_key_value_pos(json, key, vpos)) return false;
    skip_ws(json, vpos);
    return parse_json_string(json, vpos, out);
}

bool extract_int(const std::string& json, const std::string& key, int& out) {
    size_t vpos = 0; if (!find_key_value_pos(json, key, vpos)) return false;
    skip_ws(json, vpos);
    char* endp = nullptr;
    out = (int)strtol(json.c_str() + vpos, &endp, 10);
    return endp != json.c_str() + vpos;
}

bool extract_float_array(const std::string& json, const std::string& key, std::vector<float>& out) {
    size_t vpos = 0; if (!find_key_value_pos(json, key, vpos)) return false;
    skip_ws(json, vpos);
    if (vpos >= json.size() || json[vpos] != '[') return false;
    ++vpos;
    out.clear();
    while (vpos < json.size()) {
        skip_ws(json, vpos);
        if (vpos < json.size() && json[vpos] == ']') { ++vpos; break; }
        char* endp = nullptr;
        float v = strtof(json.c_str() + vpos, &endp);
        if (endp == json.c_str() + vpos) return false;
        out.push_back(v);
        vpos = (size_t)(endp - json.c_str());
        skip_ws(json, vpos);
        if (vpos < json.size() && json[vpos] == ',') { ++vpos; continue; }
        if (vpos < json.size() && json[vpos] == ']') { ++vpos; break; }
    }
    return true;
}

}

