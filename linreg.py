import numpy as np
import statsmodels.api as sm


class ModelAdapter:
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, arg):
        return self.model.predict(sm.add_constant(arg))
    
    def get_model(self):
        return self.model
    
    def get_robust_cov_model(self, cov_type='HC1'):
        if isinstance(self.model.model, sm.OLS):
            return self.model.get_robustcov_results(cov_type=cov_type)
        elif isinstance(self.model.model, sm.QuantReg | sm.RLM):
            return self.model
        else:
            raise TypeError(
                f'models of type {type(self.model)} are not supported '
                'in method ModelAdapter.get_robust_cov_model'
            )


class LogisticMLE(sm.robust.norms.RobustNorm):
    
    def __init__(self):
        super().__init__()
    
    def rho(self, z):
        return np.log(np.cosh(0.5 * z))
    
    def psi(self, z):
        return 0.5 * np.tanh(0.5 * z)
    
    def weights(self, z):
        return 0.5 * np.tanh(0.5 * z) / z
    
    def psi_deriv(self, z):
        ch = np.cosh(z)
        return 1 / (ch * ch)


class OLSModelAdapterFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return ModelAdapter(
            sm.OLS(Y, sm.add_constant(X)).fit()
        )


class LADModelAdapterFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return ModelAdapter(
            sm.QuantReg(Y, sm.add_constant(X)).fit()
        )


class LogModelAdapterFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return ModelAdapter(
            sm.RLM(Y, sm.add_constant(X), M=LogisticMLE()).fit()
        )


class HuberModelAdapterFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return ModelAdapter(
            sm.RLM(Y, sm.add_constant(X), M=sm.robust.norms.HuberT()).fit()
        )


class TukeyModelAdapterFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return ModelAdapter(
            sm.RLM(Y, sm.add_constant(X), M=sm.robust.norms.TukeyBiweight()).fit()
        )
