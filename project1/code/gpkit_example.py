#!/usr/bin/env python3

from gpkit import Variable, VectorVariable, Model
from gpkit.nomials import Monomial, Posynomial, PosynomialInequality

x = Variable("x")
y = Variable("y")
z = Variable("z")
S = 200
objective = 1/(x*y*z)
constraints = [2*x*y + 2*x*z + 2*y*z <= S,
               x >= 2*y]
m = Model(objective, constraints)
sol = m.solve(verbosity=0)
print(sol.table())
