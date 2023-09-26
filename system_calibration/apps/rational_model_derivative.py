from sympy import symbols, simplify

# Define the symbols
x, y, k1, k2, k3, k4, k5, k6 = symbols('x y k1 k2 k3 k4 k5 k6')

# Define r^2 as x^2 + y^2
r_squared = x**2 + y**2

# Define the function g
numerator = 1 + (k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3)
denominator = 1 + (k4 * r_squared + k5 * r_squared**2 + k6 * r_squared**3)
g = simplify(numerator / denominator)

# Display the result
print("g =", g)
# You can then follow the similar steps from the previous answer to find the derivatives with respect to the distortion coefficients and pixel coordinates.
dg_dk1 = g.diff(k1)
dg_dk2 = g.diff(k2)
dg_dk3 = g.diff(k3)
dg_dk4 = g.diff(k4)
dg_dk5 = g.diff(k5)
dg_dk6 = g.diff(k6)
dg_dx = g.diff(x)
dg_dy = g.diff(y)

# Display the results
print("Partial derivative of g with respect to k1:", dg_dk1)
print("Partial derivative of g with respect to k2:", dg_dk2)
print("Partial derivative of g with respect to k3:", dg_dk3)
print("Partial derivative of g with respect to k4:", dg_dk4)
print("Partial derivative of g with respect to k5:", dg_dk5)
print("Partial derivative of g with respect to k6:", dg_dk6)
print("Partial derivative of g with respect to x:", dg_dx)
print("Partial derivative of g with respect to y:", dg_dy)






