# pyright: reportMissingImports=false
#!/usr/bin/env python3
from pydantic import BaseModel, EmailStr, ValidationError


class TestEmail(BaseModel):
    email: EmailStr


# Test valid email
test_model = TestEmail(email="test@example.com")
print(f"Valid email: {test_model.email}")

# Test invalid email
try:
    TestEmail(email="invalid-email")
    print("ERROR: Invalid email was accepted")
except ValidationError as e:
    print(f"Invalid email correctly rejected: {e}")

print("EmailStr validation works!")
