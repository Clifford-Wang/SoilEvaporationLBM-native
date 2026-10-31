#pragma once
#include <stdexcept>
#include <string>

namespace trial_guard {

class TrialViolation final : public std::runtime_error {
public:
    explicit TrialViolation(const std::string& message):std::runtime_error(message){}
};

void initialize_or_throw();
void heartbeat_or_throw();
const char* contact_email();

} // namespace trial_guard
