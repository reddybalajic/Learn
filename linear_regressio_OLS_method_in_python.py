import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
# os.chdir('/home/ubuntu/PycharmProjects/Linear-Regression-(OLS-Method)-Total-Sales-Prediction')

# R code on R sample dataset

#> anova(with(ChickWeight, lm(weight ~ Time + Diet)))
#Analysis of Variance Table
#
#Response: weight
#           Df  Sum Sq Mean Sq  F value    Pr(>F)
#Time        1 2042344 2042344 1576.460 < 2.2e-16 ***
#Diet        3  129876   43292   33.417 < 2.2e-16 ***
#Residuals 573  742336    1296
#write.csv(file='ChickWeight.csv', x=ChickWeight, row.names=F)

cw = pd.read_csv('Advertising.csv')
y=cw.Sales
X=cw.TV

X = sm.add_constant(X)  # Adds a constant term to the predictor
# print X.head()

est = sm.OLS(y, X)
est = est.fit()
print(est.summary())

est.params

X_prime = np.linspace(X.TV.min(), X.TV.max(), 100)[:, np.newaxis]
X_prime = sm.add_constant(X_prime)  # add constant as we did before

y_hat = est.predict(X_prime)

plt.scatter(X.TV, y, alpha=0.3)  # Plot the raw data
plt.xlabel("TV in number")
plt.ylabel("Total Sales")
plt.plot(X_prime[:, 1], y_hat, 'r', alpha=0.9)  # Add the regression line, colored in red
plt.show()




# cw_lm=ols('Sales ~ TV', data=cw).fit() #Specify C for Categorical
# print cw_lm.summary()

#print(sm.stats.anova_lm(cw_lm, typ=2))
#                  sum_sq   df            F         PR(>F)
#C(Diet)    129876.056995    3    33.416570   6.473189e-20
#Time      2016357.148493    1  1556.400956  1.803038e-165
#Residual   742336.119560  573          NaN            NaN