import numpy as np
import scipy.stats as stats


class Option():
    def __init__(self, stock, strike, risk_free, sigma, maturity, dividend=0, implied = True):
        self.stock = stock
        self.strike = strike
        self.risk_free = risk_free
        self.sigma = sigma
        self.original_sigma = sigma
        self.maturity = maturity
        self.dividend = dividend
        self.implied = implied
        self.lamda = 1.0
        if self.implied:
            sigma = sigma * np.exp(-self.lamda * stock/strike)
            self.sigma = sigma
        self.d1 = (np.log(stock/strike)+(risk_free-dividend+sigma**2/2)*maturity)/(sigma*np.sqrt(maturity))
        self.d2 = (np.log(stock/strike)+(risk_free-dividend-sigma**2/2)*maturity)/(sigma*np.sqrt(maturity))

        self.call_price = stock*np.exp(-dividend*maturity)*stats.norm.cdf(self.d1)-strike*stats.norm.cdf(self.d2)*np.exp(-risk_free*maturity)
        self.put_price = strike*stats.norm.cdf(-self.d2)*np.exp(-risk_free*maturity)-stock*np.exp(-dividend*maturity)*stats.norm.cdf(-self.d1)

    def get_call(self):
        print("The put price is:", self.call_price)
        return self.call_price
    
    def get_put(self):
        print("The put price is:", self.put_price)
        return self.put_price
    
    def get_delta(self, beta=1.0):
        #first derivative with respect to S equal to 0
        delta_call = beta*stats.norm.cdf(self.d1)*np.exp(-self.dividend*self.maturity) #it is actually the derivative of the option with respect to the underlying
        delta_put = beta*(stats.norm.cdf(self.d1)-1)*np.exp(-self.dividend*self.maturity)
        deriv_call = 1.0 #compute derivative of call price with respect to sigma_implied
        exp_func = 1.0/np.sqrt(2*np.pi)*np.exp(-self.d1**2/2)
        exp_func2 = 1.0/np.sqrt(2*np.pi)*np.exp(-self.d2**2/2)
        if self.implied:
            denominator1 = -1.0/self.sigma + self.sigma*self.maturity/(self.stock/self.strike+(self.risk_free-self.dividend+self.sigma**2/2)*self.maturity)
            denominator2 = -1.0/self.sigma - self.sigma*self.maturity/(self.stock/self.strike+(self.risk_free-self.dividend-self.sigma**2/2)*self.maturity)
            delta_call+= -self.original_sigma*self.lamda/self.strike*(self.stock*np.exp(-self.dividend*self.maturity)*exp_func*denominator1-self.strike*exp_func2*np.exp(self.risk_free*self.maturity*denominator2))
            delta_put += -self.original_sigma*self.lamda/self.strike*(-self.strike*exp_func2*np.exp(-self.risk_free*self.maturity*denominator2+self.stock*np.exp(self.dividend*self.maturity)*exp_func*denominator1))
        print("The number of shares needed to hedge a call option are:", delta_call)
        print("The number of shares needed to hedge a put option are:", delta_put)
        return delta_call, delta_put
    def get_theta(self, beta=1.0):
        #first derivative with respect to t equal to 0
        exp_func = 1.0/np.sqrt(2*np.pi)*np.exp(-self.d1**2/2)
        theta_call = -self.stock*np.exp(-self.dividend*self.maturity)*self.sigma*exp_func/2.0/np.sqrt(self.maturity)
        theta_call+=-self.risk_free*self.strike*np.exp(-self.risk_free*self.maturity)*stats.norm.cdf(-self.d2)
        theta_call+= self.dividend*self.stock*np.exp(-self.dividend*self.maturity)*stats.norm.cdf(self.d1)

        theta_put = -self.stock*np.exp(-self.dividend*self.maturity)*self.sigma*exp_func/2.0/np.sqrt(self.maturity)
        theta_put += self.risk_free*self.strike*np.exp(-self.risk_free*self.maturity)*stats.norm.cdf(self.d2)
        theta_put+= -self.dividend*self.stock*np.exp(-self.dividend*self.maturity)*stats.norm.cdf(-self.d1)
        return beta*theta_call, beta*theta_put
    def get_gamma(self, beta = 1.0):
        #second derivative with respect to S equal to 0
        exp_func = 1.0/np.sqrt(2*np.pi)*np.exp(-self.d1**2/2)
        gamma = exp_func/self.stock/self.sigma/np.sqrt(self.maturity)*np.exp(-self.dividend*self.maturity)
        return gamma, gamma
    def get_vega(self, beta = 1.0):
        #first derivative with respect to sigma equal to 0
        exp_func = 1.0/np.sqrt(2*np.pi)*np.exp(-self.d1**2/2)
        vega = self.stock*np.sqrt(self.maturity)*exp_func*np.exp(-self.dividend*self.maturity)
        return vega, vega
    def geta_rho(self, beta = 1.0):
        #first derivative with respect to r equal to 0
        rho_call = self.strike*self.maturity*np.exp(-self.risk_free*self.maturity)*stats.norm.cdf(self.d2)
        rho_put = -self.strike*self.maturity*np.exp(self.risk_free*self.maturity)**stats.norm.cdf(-self.d2)
        return rho_call, rho_put
