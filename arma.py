from statsmodels.tsa.arima.model import ARIMA


def estimate_ARMA(series, max_p, max_q, print_progress=False):
    
    opt_model = ARIMA(series, order=(0, 0, 1)).fit()
    
    if print_progress:
        print(f'ARMA(0, 1) BIC = {opt_model.bic:.3f}')
    
    for p in range(0, max_p + 1):
        
        for q in range(0, max_q + 1):
            
            if p == 0 and q in {0, 1}:
                continue
            
            model = ARIMA(series, order = (p, 0, q)).fit()
            if model.bic < opt_model.bic:
                opt_model = model
            
            if print_progress:
                print(f'ARMA({p}, {q}) BIC = {model.bic:.3f}')
    
    return opt_model
