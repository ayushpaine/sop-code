import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline
 
# Our time-dependent matrix
def A(t):
  return np.array([[np.sin(t), np.cos(t)], [-(np.cos(t)), np.sin(t)]])
 
# Value of our ZNN parameter
gamma = 1000
 
def equation(t, state):
    x11, x12, x21, x22 = state
 
    # The following equations come after simplifying the initial ZNN equation 
    #E'(t) = -γ*Φ(E(t)) where E is our Zhang Function A(t)*X(t) - I
 
    dx11 = gamma*np.sin(t)  + x21 - gamma*x11
    dx12 = -1*gamma*np.cos(t) + x22 - gamma*x12
    dx21 = gamma*np.cos(t) - x11 - gamma*x21
    dx22 = gamma*np.sin(t) - x12 - gamma*x22
 
    return [dx11, dx12, dx21, dx22]
 
#initial value of the X(t) matrix (calculated inverse at each point)
y0 = [0, 0, 0, 0]
 
t_span_graph = (0.0, 100.0) #use this for the entire time bouund
times_graph = np.linspace(t_span_graph[0], t_span_graph[1], 500)
 
t = math.pi/2
t_span_fixed = (t, t + 0.01)
times_fixed = np.linspace(t_span_fixed[0], t_span_fixed[1], 1)
 
result_solve_ivp_graph = solve_ivp(equation, t_span_graph, y0)
result_solve_ivp_fixed = solve_ivp(equation, t_span_fixed, y0)
 
# Tuple of outputs for each of the element the associated X(t) 
#matrix and times at which it's calculated
time = result_solve_ivp_graph.t
ans11_graph = result_solve_ivp_graph.y[0]
ans12_graph = result_solve_ivp_graph.y[1]
ans21_graph = result_solve_ivp_graph.y[2]
ans22_graph = result_solve_ivp_graph.y[3]

ans11_fixed = result_solve_ivp_fixed.y[0]
ans12_fixed = result_solve_ivp_fixed.y[1]
ans21_fixed = result_solve_ivp_fixed.y[2]
ans22_fixed = result_solve_ivp_fixed.y[3]
 
A_t = []
 
# Values of A for the given timestamps
for i in time:
  A_t.append(A(i))
 
x11_t = []
x12_t = []
x21_t = []
x22_t = []
 
# Values of each of the elements of A
for i in A_t:
  x11_t.append(i[0][0])
  x12_t.append(i[0][1])
  x21_t.append(i[1][0])
  x22_t.append(i[1][1])
 
error = []
 
range = np.arange(0, 50)

if len(range) > 1:
  for i in range:
    # Frobenius norm of the error function A(t)*X(t) - I
    error.append((
        (x11_t[i]*ans11_graph[i] + x12_t[i]*ans21_graph[i] - 1)**2 + 
        (x11_t[i]*ans12_graph[i] + x12_t[i]*ans22_graph[i])**2 + 
        (x21_t[i]*ans11_graph[i] + x22_t[i]*ans21_graph[i])**2 + 
        (x21_t[i]*ans12_graph[i] + x22_t[i]*ans22_graph[i] - 1)**2)**0.5)
 
time_bound = 1
plot_time = np.linspace(0, time_bound, 5*time_bound)

# Converts the graph from a discrete one to continuous form using interpolation
X_Y_Spline = make_interp_spline(plot_time, error[0: 5*time_bound])
X_ = np.linspace(plot_time.min(), plot_time.max(), 500)
Y_ = X_Y_Spline(X_)
 
# Plotting the Graph
# plt.plot(X_, Y_)

ans = [[ans11_fixed[-1], ans12_fixed[-1]],[ans21_fixed[-1], ans22_fixed[-1]]]
 
ans = np.array(ans)

print(ans)

