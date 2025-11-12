#pragma once

#include <string>
#include <vector>

namespace minijson {

// Minimal JSON helpers for simple key lookup and extraction.

// Escape a string for JSON string literal.
std::string escape(const std::string& s);

// Extract a string field from a flat JSON object: {"key":"value", ...}
bool extract_string(const std::string& json, const std::string& key, std::string& out);

// Extract an integer field from a flat JSON object: {"key":123, ...}
bool extract_int(const std::string& json, const std::string& key, int& out);

// Extract a float array field: {"key":[1.0,2.0,...]}
bool extract_float_array(const std::string& json, const std::string& key, std::vector<float>& out);

}

