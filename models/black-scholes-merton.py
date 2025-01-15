import math
from scipy.stats import norm

#risk free interest rate r
class BlackScholesMerton:
    def __init__(self, S0, strike_price, maturity, r, sigma, div):
        self.S0 = S0
        self.strike_price = strike_price
        self.maturity = maturity
        self.r = r
        self.sigma = sigma
        self.div = div
        self.d1 = ((math.log(self.S0 / self.strike_price) + (self.r * self.maturity)) / (self.sigma * math.sqrt(self.maturity))) + ((self.sigma * math.sqrt(self.maturity)) / 2)
        self.d2 = self.d1 - (self.sigma * math.sqrt(self.maturity))

    def call(self):
        call = (self.S0 * math.exp(-self.div * self.maturity) * norm.cdf(self.d1)) - (self.strike_price * math.exp(-self.r * self.maturity) * norm.cdf(self.d2))
        return call
    
    def put(self):
        put = (self.strike_price * math.exp(-self.r * self.maturity) * norm.cdf(-self.d2)) - (self.S0 * math.exp(-self.div * self.maturity) * norm.cdf(-self.d1))
        return put
    
test = BlackScholesMerton(60, 50, 4/12, 0.03, 0.375, 0)
print("Call: ", test.call())
print("Put", test.put())