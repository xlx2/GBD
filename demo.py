import cvxpy as cp

# 定义变量
x = cp.Variable()

# 定义目标函数
objective = cp.Minimize(x**2)

# 定义约束
constraints = [0 >= 1 - cp.sqrt(x)]

# 定义优化问题
problem = cp.Problem(objective, constraints)

# 求解问题
problem.solve()

# 获取拉格朗日乘子
lagrange_multiplier = constraints[0].dual_value
print("拉格朗日乘子:", lagrange_multiplier)
