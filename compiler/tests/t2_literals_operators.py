# every literal shape and operator the lexer must recognize
i = 42
f1 = 3.14
f2 = .5
f3 = 1e6
f4 = 2.5e-3
s1 = "double"
s2 = 'single'
s3 = """triple
spanning lines"""
a = 1 + 2 - 3 * 4 / 5 // 6 % 7 ** 8
b = (1 < 2) <= (3 > 4) >= (5 == 6) != (7 is not 8) and 9 in x or not y
c = p & q | r ^ ~s << 2 >> 1
a += 1; a -= 2; a *= 3; a /= 4; a //= 5; a %= 6; a **= 7
a &= 1; a |= 2; a ^= 3; a <<= 4; a >>= 5
m = w[:, None] @ w[None, :]
def g(x) -> int: return x
t = (1,); l = [1, 2]; d = {"k": 1}
e = ...
