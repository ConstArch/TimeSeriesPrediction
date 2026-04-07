from abc         import ABC, abstractmethod
from enum        import auto as enum_auto, Enum
from dataclasses import dataclass
from typing      import Optional

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools


class AbstractTestResult(ABC):
    
    @abstractmethod
    def to_str(self, indentation=1):
        pass
    
    def __str__(self):
        return self.to_str()
    
    @abstractmethod
    def evaluate(self, significance_level=0.05):
        pass
    
    @abstractmethod
    def to_str_evaluated(self, *, significance_level=0.05, indentation=1):
        pass


class AutoLagInfoCriterionType(Enum):
    
    AIC   = enum_auto()
    BIC   = enum_auto()
    TSTAT = enum_auto()
    NONE  = enum_auto()
    
    def to_adfuller_parameter(self):
        if self is AutoLagInfoCriterionType.AIC:
            return 'AIC'
        elif self is AutoLagInfoCriterionType.BIC:
            return 'BIC'
        elif self is AutoLagInfoCriterionType.TSTAT:
            return 't-stat'
        else:
            return None


@dataclass
class AutoLagResult:
    
    result_lags    : int
    max_lag        : int
    max_lag_auto   : bool
    info_crit      : AutoLagInfoCriterionType
    info_crit_best : Optional[float] = None
    
    def to_str(self, indentation=1):
        ind_str = '    ' * indentation
        if self.info_crit is not AutoLagInfoCriterionType.NONE:
            return (
                f'info criterion:\n'
                f'{ind_str}result lag     = {self.result_lags}\n'
                f'{ind_str}max lag        = {self.max_lag}{" (automatic estimation)" if self.max_lag_auto else ""}\n'
                f'{ind_str}info criterion = {self.info_crit} (best={self.info_crit_best:.3f})'
            )
        else:
            return (
                f'without info criterion:\n'
                f'{ind_str}result lag = {self.result_lags}\n'
                f'{ind_str}max lag    = {self.max_lag}{" (automatic estimation)" if self.max_lag_auto else ""}'
            )
    
    def __str__(self):
        return self.to_str()


class DickeyFullerDistributionType(Enum):
    NONE      = enum_auto()
    CONSTANT  = enum_auto()
    LINEAR    = enum_auto()
    QUADRATIC = enum_auto()


@dataclass
class AugmentedDickeyFullerTestResult(AbstractTestResult):
    
    distribution       : DickeyFullerDistributionType
    ADF_stat           : float
    pvalue             : float
    series_obs         : int
    ADF_regression_obs : int
    auto_lag_result    : AutoLagResult
    
    def to_str(self, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'ADF test result:\n'
            f'{ind_str}test statistic distribution        = {self.distribution}\n'
            f'{ind_str}test statistic value               = {self.ADF_stat:.3f}\n'
            f'{ind_str}p-value                            = {self.pvalue:.3f}\n'
            f'{ind_str}series observations number         = {self.series_obs}\n'
            f'{ind_str}ADF regression observations number = {self.ADF_regression_obs}\n'
            f'{ind_str}used lags = {self.auto_lag_result.to_str(indentation + 1)}'
        )
    
    def evaluate(self, significance_level=0.05):
        return 'TSP' if self.pvalue < significance_level else 'DSP'
    
    def to_str_evaluated(self, *, significance_level=0.05, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'{self.to_str(indentation)}\n'
            f'{ind_str}evaluated = evaluated:\n'
            f'{ind_str}    solution           = {self.evaluate(significance_level)}\n'
            f'{ind_str}    significance level = {significance_level}'
        )


class TrendType(Enum):
    
    NONE      = enum_auto()
    CONSTANT  = enum_auto()
    LINEAR    = enum_auto()
    QUADRATIC = enum_auto()
    
    def to_adfuller_parameter(self):
        if self is TrendType.NONE:
            return 'n'
        elif self is TrendType.CONSTANT:
            return 'c'
        elif self is TrendType.LINEAR:
            return 'ct'
        else:
            return 'ctt'
    
    def to_DF_distribution_type(self):
        if self is TrendType.NONE:
            return DickeyFullerDistributionType.NONE
        elif self is TrendType.CONSTANT:
            return DickeyFullerDistributionType.CONSTANT
        elif self is TrendType.LINEAR:
            return DickeyFullerDistributionType.LINEAR
        else:
            return DickeyFullerDistributionType.QUADRATIC


def augmented_Dickey_Fuller_test(series, trend_type, max_lag=None, auto_lag_IC=AutoLagInfoCriterionType.AIC):
    
    ADF_stat, pvalue, crit_values, results_store = statsmodels.tsa.stattools.adfuller(
        series,
        regression=trend_type.to_adfuller_parameter(),
        maxlag=max_lag,
        autolag=auto_lag_IC.to_adfuller_parameter(),
        store=True
    )
    
    auto_lag_result = AutoLagResult(
        result_lags    = results_store.usedlag,
        max_lag        = results_store.maxlag,
        max_lag_auto   = max_lag is None,
        info_crit      = auto_lag_IC,
        info_crit_best = results_store.icbest
    )
    
    return AugmentedDickeyFullerTestResult(
        distribution       = trend_type.to_DF_distribution_type(),
        ADF_stat           = ADF_stat,
        pvalue             = pvalue,
        series_obs         = len(series),
        ADF_regression_obs = results_store.nobs,
        auto_lag_result    = auto_lag_result
    )


@dataclass
class CoefficientSignificanceTestResult(AbstractTestResult):
    
    distribution   : str
    statistic      : float
    pvalue         : float
    sample_size    : int
    
    def to_str(self, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'coefficient significance test result:\n'
            f'{ind_str}test statistic distribution = {self.distribution}\n'
            f'{ind_str}test statistic value        = {self.statistic:.3f}\n'
            f'{ind_str}p-value                     = {self.pvalue:.3f}\n'
            f'{ind_str}sample size                 = {self.sample_size}'
        )
    
    def evaluate(self, significance_level=0.05):
        return 'significant' if self.pvalue < significance_level else 'not significant'
    
    def to_str_evaluated(self, *, significance_level=0.05, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'{self.to_str(indentation)}\n'
            f'{ind_str}evaluated = evaluated:\n'
            f'{ind_str}    solution           = {self.evaluate(significance_level)}\n'
            f'{ind_str}    significance level = {significance_level}'
        )


def linear_trend_significance_Student_test(series, linreg_model_adapter_factory):
    
    n = len(series)
    model = linreg_model_adapter_factory(np.arange(n), series).model
    
    return CoefficientSignificanceTestResult(
        distribution = f'St({n - 2})',
        statistic    = model.tvalues[1],
        pvalue       = model.pvalues[1],
        sample_size  = n
    )


def constant_bias_significance_Student_test(series, linreg_model_adapter_factory):
    
    n = len(series)
    model = linreg_model_adapter_factory(np.ones(n), series).model
    
    return CoefficientSignificanceTestResult(
        distribution = f'St({n - 1})',
        statistic    = model.tvalues[0],
        pvalue       = model.pvalues[0],
        sample_size  = n
    )


@dataclass
class DoladoJenkinsonSosvillaRiveroProcedureResult(AbstractTestResult):
    
    ADF_linear   : AugmentedDickeyFullerTestResult
    ADF_constant : AugmentedDickeyFullerTestResult
    ADF_none     : AugmentedDickeyFullerTestResult
    
    St_linear   : CoefficientSignificanceTestResult
    St_constant : CoefficientSignificanceTestResult
    
    def to_str(self, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'Dolado --- Jenkinson --- Sosvilla-Rivero procedure result:\n\n'
            f'{ind_str}augmented DF for linear trend test result = {     self.ADF_linear  .to_str_evaluated(indentation=indentation+1)}\n\n'
            f'{ind_str}linear trend significance Student test result = { self.St_linear   .to_str_evaluated(indentation=indentation+1)}\n\n'
            f'{ind_str}augmented DF for constant bias test result = {    self.ADF_constant.to_str_evaluated(indentation=indentation+1)}\n\n'
            f'{ind_str}constant bias significance Student test result = {self.St_constant .to_str_evaluated(indentation=indentation+1)}\n\n'
            f'{ind_str}augmented DF for zero mean test result = {        self.ADF_none    .to_str_evaluated(indentation=indentation+1)}\n'
        )
    
    def evaluate(self, significance_level=0.05):
        
        if self.ADF_linear.pvalue < significance_level:
            
            #if St_linear.pvalue < significance_level:
            #    return 'TSP & linear trend'
            #elif St_constant.pvalue < significance_level:
            #    return 'TSP & constant bias'
            #else:
            #    return 'TSP & zero mean'
            return 'TSP & linear trend'
            
        elif self.St_linear.pvalue < significance_level:
            
            return 'DSP & linear trend'
            
        elif self.ADF_constant.pvalue < significance_level:
            
            #if St_constant.pvalue < significance_level:
            #    return 'TSP & constant bias'
            #else:
            #    return 'TSP & zero mean'
            return 'TSP & constant bias'
            
        elif self.St_constant.pvalue < significance_level:
            
            return 'DSP & constant bias'
            
        elif self.ADF_none.pvalue < significance_level:
            
            return 'TSP & zero mean'
            
        else:
            
            return 'DSP & zero mean'
        
    # end evaluate
    
    def to_str_evaluated(self, *, significance_level=0.05, indentation=1):
        ind_str = '    ' * indentation
        return (
            f'{self.to_str(indentation)}\n'
            f'{ind_str}evaluated = evaluated:\n'
            f'{ind_str}    solution           = {self.evaluate(significance_level)}\n'
            f'{ind_str}    significance level = {significance_level}'
        )


def Dolado_Jenkinson_SosvillaRivero_procedure(
    series,
    linreg_model_adapter_factory,
    ADF_max_lag=None,
    ADF_auto_lag_IC=AutoLagInfoCriterionType.AIC
):
    delta_series = np.diff(series)
    
    return DoladoJenkinsonSosvillaRiveroProcedureResult(
        ADF_linear   = augmented_Dickey_Fuller_test(series, TrendType.LINEAR  , max_lag=ADF_max_lag, auto_lag_IC=ADF_auto_lag_IC),
        ADF_constant = augmented_Dickey_Fuller_test(series, TrendType.CONSTANT, max_lag=ADF_max_lag, auto_lag_IC=ADF_auto_lag_IC),
        ADF_none     = augmented_Dickey_Fuller_test(series, TrendType.NONE    , max_lag=ADF_max_lag, auto_lag_IC=ADF_auto_lag_IC),
        St_linear    =  linear_trend_significance_Student_test(delta_series, linreg_model_adapter_factory),
        St_constant  = constant_bias_significance_Student_test(delta_series, linreg_model_adapter_factory)
    )


#def Dolado_Jenkinson_SosvillaRivero_procedure(X, significance_level=0.05):
#    
#    pvalue_ADF_ct = statsmodels.tsa.stattools.adfuller(X, regression='ct')[1]
#    if pvalue_ADF_ct < significance_level:
#        return f'TSP + linear trend ({pvalue_ADF_ct=})'
#    
#    if sm.OLS(np.diff(X), sm.add_constant(np.arange(len(X) - 1))).fit().pvalues[1] < significance_level:
#        return 'DSP + linear trend'
#    
#    pvalue_ADF_c = statsmodels.tsa.stattools.adfuller(X, regression='c')[1]
#    if pvalue_ADF_c < significance_level:
#        return f'TSP + constant ({pvalue_ADF_c=})'
#    
#    if sm.OLS(np.diff(X), np.ones(len(X) - 1)).fit().pvalues[0] < significance_level:
#        return 'DSP + constant'
#    
#    pvalue_ADF_0 = statsmodels.tsa.stattools.adfuller(X, regression='n')[1]
#    if pvalue_ADF_0 < significance_level:
#        return 'TSP'
#    else:
#        return 'DSP'


@dataclass
class GrangerTestResult(AbstractTestResult):
    
    distribution   : str
    statistic      : float
    pvalue         : float
    series_obs     : int
    reason_max_lag : int
    effect_max_lag : int
    
    def to_str(self, indentation=0):
        ind_str = '    ' * indentation
        return (
            f'{ind_str}Granger test result:\n'
            f'{ind_str}    test statistic distribution : {self.distribution}\n'
            f'{ind_str}    test statistic value        : {self.statistic:.3f}\n'
            f'{ind_str}    p-value                     : {self.pvalue:.3f}\n'
            f'{ind_str}    series observations number  : {self.series_obs}\n'
            f'{ind_str}    reason used lags number     : {self.reason_max_lag}\n'
            f'{ind_str}    effect used lags number     : {self.reason_max_lag}'
        )
    
    def evaluate(self, significance_level=0.05):
        return 'causality' if self.pvalue < significance_level else 'non-causality'
    
    def to_str_evaluated(self, *, significance_level=0.05, indentation=0):
        ind_str = '    ' * indentation
        return (
            f'{self.to_str(indentation)}\n'
            f'{ind_str}evaluated:\n'
            f'{ind_str}    solution           : {self.evaluate(significance_level)}\n'
            f'{ind_str}    significance level : {significance_level}'
        )


def Granger_causality_robust_Wald_test(
    reason_series,
    effect_series,
    reason_max_lag,
    effect_max_lag,
    model_adapter_factory,
    cov_type='HC1',
    use_f=None
):
    series_size = len(reason_series)
    sample_size = series_size - max(reason_max_lag, effect_max_lag)
    series_dtype = reason_series.dtype
    
    reason_AR_factors_T = np.empty(shape = (reason_max_lag, sample_size), dtype = series_dtype)
    effect_AR_factors_T = np.empty(shape = (effect_max_lag, sample_size), dtype = series_dtype)
    
    for i in range(1, reason_max_lag + 1):
        reason_AR_factors_T[-i] = reason_series[-i - sample_size : -i]
    
    for i in range(1, effect_max_lag + 1):
        effect_AR_factors_T[-i] = effect_series[-i - sample_size : -i]
    
    AR_factors = np.concatenate([reason_AR_factors_T, effect_AR_factors_T]).T
    AR_targets = effect_series[effect_max_lag : effect_max_lag + sample_size]
    
    condition_matrix = np.zeros(
        shape = (reason_max_lag, 1 + reason_max_lag + effect_max_lag),
        dtype = series_dtype
    )
    
    for i in range(reason_max_lag):
        condition_matrix[i, 1 + i] = 1
    
    condition_right_part = np.zeros(shape = reason_max_lag, dtype = series_dtype)
    
    test_result = model_adapter_factory(AR_factors, AR_targets) \
        .get_robust_cov_model(cov_type) \
        .wald_test(r_matrix = (condition_matrix, condition_right_part), use_f = use_f)
    
    if test_result.distribution == 'F':
        distribution = f'F({int(test_result.df_num)}, {int(test_result.df_denom)})'
    else:
        distribution = f'chi^2({int(test_result.df_denom)})'
    
    return GrangerTestResult(
        distribution   = distribution,
        statistic      = test_result.statistic.squeeze().item(),
        pvalue         = test_result.pvalue.item(),
        series_obs     = series_size,
        reason_max_lag = reason_max_lag,
        effect_max_lag = effect_max_lag
    )
