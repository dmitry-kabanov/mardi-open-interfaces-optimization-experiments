import numpy as np
from scipy import optimize


class RawOptim:
    def __init__(self):
        self.user_data = None

    def set_initial_guess(self, x0):
        self.x0 = x0
        self.x = np.empty_like(self.x0)

    def set_objective_fn(self, objective_fn):
        self.objective_fn = objective_fn

        x = np.random.random(size=(len(self.x0),))
        msg = "Wrong signature for the objective function: "
        msg += "expected return value must be of `float64` type"

        assert type(self.objective_fn(x, self.user_data)) in [
            float,
            np.float64,
        ], msg

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn

    def set_method(self, method_name, method_params):
        available_options = optimize.show_options(
            "minimize", method=method_name, disp=False
        )

        for k in method_params.keys():
            candidates = [f"\n{k} :", f"\n{k},"]
            found = False
            for c in candidates:
                if c in available_options:
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Method option '{k}' is not known for the method '{method_name}'"
                )

        self.method_name = method_name
        self.method_params = method_params

    def minimize(self):
        grad_fn = self.grad_wrapper

        result = optimize.minimize(
            self.objective_fn,
            self.x0,
            args=(self.user_data,),
            jac=grad_fn,  # I wonder, why the gradient in scipy.optimize is `jac`?
            method=self.method_name,
            options=self.method_params,
        )

        self.x[:] = result.x
        print(result)
        if hasattr(result, "jac"):
            print("|∇f(x*)|_∞ = ", np.linalg.norm(result.jac, np.inf))
        return (result.status, result.message)

    def grad_wrapper(self, x, __):
        grad_values = np.empty_like(x)
        # grad_values = self.grad_values
        self.grad_fn(x, grad_values, self.user_data)
        return grad_values
