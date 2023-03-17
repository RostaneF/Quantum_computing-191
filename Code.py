# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:59:47 2023

@author: rosta
"""
import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf

#For Neural network : 
import pandas_datareader as pdr
import fix_yahoo_finance as fx

class Pricer : 
        def __init__(self, S0:float, K:float, r:float,  sigma:float, T:float, num_steps:int, O_type:str):
            
            #Check :
            assert S0 >= 0, 'initial stock price cannot be less than zero'
            assert T >= 0, 'time to maturity cannot be less than zero'
            assert O_type in ["Call","Put"], 'The priced Option can be either a Call or a Put'
            
            
            #Model parameters
            self.S0 = S0 
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.num_steps = num_steps
            self.Option_Type = O_type
            
            #For simulations :
            self.S_vec = None
            self.price_vec = None
            self.P_matrix = None
            
            #Neural network :
            self.price = None
            self.S = None
            self.V = None
            self.Payoff = None
            self.Mu = None
            self.N = 10000
            
            
######################### ######################### ######################### ######################### ######################### ######################### ######################### 
######################### #########################  CRR model & American option pricing ######################### ######################### ######################### 
######################### ######################### ######################### ######################### ######################### #########################          
                  
        
        def CRR_model(self):
            # We divide the time interval [0, T] into N steps of length h
            h = self.T/self.num_steps
            
            # And we set the parameters of the binomial model to be :
            u = np.exp(self.sigma*np.sqrt(h))
            d = np.exp(-self.sigma*np.sqrt(h))
            q = (np.exp(self.r*h)-d)/(u-d) #Risk neutral probability 
            discount = np.exp(-self.r*h)
            
            Binomial_Tree = [[0 for i in range(t+1)] for t in range(self.num_steps+1)]
            Binomial_Tree[0][0] = self.S0
            
            Payoff_Tree = [[0 for i in range(t+1)] for t in range(self.num_steps+1)]
            
            
            #Fill Binomial Tree : 
            for Node in Binomial_Tree[1:]:
                j = 1
                for i in range(len(Node)): 
                    power_u = len(Node) - j
                    power_d = j - 1 
                    Node[i] = self.S0*(u**power_u)*(d**power_d)
                    j+=1 
            
            
            #Fill CRR_Payoff_Tree : 
            Payoff_Tree[self.num_steps] = [payoff(Binomial_Tree[self.num_steps][j], self.K, self.Option_Type) for j in range(len(Binomial_Tree[self.num_steps]))]
            for i in range(len(Payoff_Tree)-2,-1,-1):
                for j in range(len(Payoff_Tree[i])):
                    Payoff_Tree[i][j] = (q*Payoff_Tree[i+1][j] + (1-q)*Payoff_Tree[i+1][j+1])*discount
            
            # We can return Binomial_Tree,Payoff_Tree
            
            return Payoff_Tree[0][0]
        
        def payoff_f(self, S):
            if self.Option_Type == "Call":
                Payoff = np.maximum( S - self.K, 0 )
            elif self.Option_Type == "Put":    
                Payoff = np.maximum( self.K - S, 0 )  
            return Payoff
        
        def American_Price(self):
            # We divide the time interval [0, T] into N steps of length h
            h = self.T/self.num_steps
            
            # And we set the parameters of the binomial model to be :
            '''  
            U = np.exp((self.r + 0.5*self.sigma**2)*h + self.sigma*np.sqrt(h)) - 1 # Up
            D = np.exp((self.r + 0.5*self.sigma**2)*h - self.sigma*np.sqrt(h)) - 1  #Down
            R = np.exp(self.r*h) - 1 # Such as 1+r = exp(rh)
            Q = (R-D)/(U-D) #Risk neutral probability 
            discount = 1/(1+R) #'''
            
            U = np.exp(self.sigma*np.sqrt(h))
            D = np.exp(-self.sigma*np.sqrt(h))
            Q = (np.exp(self.r*h)-D)/(U-D) #Risk neutral probability 
            discount = np.exp(-self.r*h)
            
            American_Tree = [[0 for i in range(t+1)] for t in range(self.num_steps+1)]
            American_Tree[0][0] = self.S0
            
            Payoff_Tree = [[0 for i in range(t+1)] for t in range(self.num_steps+1)]
            
            
            #Fill American Tree : 
            for Node in American_Tree[1:]:
                j = 1
                for i in range(len(Node)): 
                    power_u = len(Node) - j
                    power_d = j - 1 
                    Node[i] = self.S0*(U**power_u)*(D**power_d)
                    j+=1 
                    
            Payoff_Tree[self.num_steps] = [payoff(American_Tree[self.num_steps][j], self.K, self.Option_Type) for j in range(len(American_Tree[self.num_steps]))]
            
            #Fill American Payoff_Tree : 
            for i in range(len(Payoff_Tree)-2,-1,-1):
                for j in range(len(Payoff_Tree[i])):
                    Payoff_Tree[i][j] = max(payoff(American_Tree[i][j],self.K,self.Option_Type) ,(Q*Payoff_Tree[i+1][j] + (1-Q)*Payoff_Tree[i+1][j+1])*discount)

            #return American_Tree,Payoff_Tree  
            return Payoff_Tree[0][0]
    
######################### ######################### ######################### ######################### ######################### ######################### ######################### 
######################### #########################  PDE Option pricing (Partial differential equations) ######################### ######################### ######################### 
######################### ######################### ######################### ######################### ######################### #########################          

        
        def PDE_price(self, Nspace = 10000, exercise = "American", Time=False, solver="splu"):
            """
            steps = tuple with number of space steps and time steps
            payoff = "call" or "put"
            exercise = "European" or "American"
            Time = Boolean. Execution time.
            Solver = spsolve or splu or Thomas or SOR
            """
            t_init = time.time()
            
            Ntime = self.num_steps
            
            S_max = 6*float(self.K)                
            S_min = float(self.K)/6
            x_max = np.log(S_max)
            x_min = np.log(S_min)
            x0 = np.log(self.S0)                            # current log-price
            
            x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  
            t, dt = np.linspace(0, self.T, Ntime, retstep=True)
            
            self.S_vec = np.exp(x)        # vector of S
            Payoff = self.payoff_f(self.S_vec)
    
            V = np.zeros((Nspace,Ntime))
            if self.Option_Type == "Call":
                V[:,-1] = Payoff
                V[-1,:] = Payoff[len(Payoff)-1] * np.exp(-self.r* t[::-1] )
                V[0,:]  = 0
            else:    
                V[:,-1] = Payoff
                V[-1,:] = 0
                V[0,:]  = Payoff[0] * np.exp(-self.r* t[::-1] )    # Instead of Payoff[0] I could use K 
                                                        # For s to 0, the limiting value is e^(-rT)(K-s)     
            
            sig2 = self.sigma**2 
            dxx = dx**2
            a = ( (dt/2) * ( (self.r-0.5*sig2)/dx - sig2/dxx ) )
            b = ( 1 + dt * ( sig2/dxx + self.r ) )
            c = (-(dt/2) * ( (self.r-0.5*sig2)/dx + sig2/dxx ) )
            
            D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()
                
            offset = np.zeros(Nspace-2)
            
            
            if solver == "spsolve": # Si le solver est "spsolve", on utilise la fonction spsolve pour résoudre le système linéaire
                if exercise=="European": 
                    for i in range(Ntime-2,-1,-1): 
                        offset[0] = a * V[0,i] # On calcule le terme d'offset pour le premier nœud de la grille
                        offset[-1] = c * V[-1,i] # On calcule le terme d'offset pour le dernier nœud de la grille
                        # On a donc construit le B(Boundary term) qui correspond à [a*Vo^n, ..., c*V_M^n]
                        V[1:-1,i] = spsolve( D, (V[1:-1,i+1] - offset) ) # On résout le système linéaire pour chaque nœud de la grille en utilisant spsolve
                elif exercise=="American": 
                    for i in range(Ntime-2,-1,-1): 
                        offset[0] = a * V[0,i] # On calcule le terme d'offset pour le premier nœud de la grille
                        offset[-1] = c * V[-1,i] # On calcule le terme d'offset pour le dernier nœud de la grille
                        V[1:-1,i] = np.maximum( spsolve( D, (V[1:-1,i+1] - offset) ), Payoff[1:-1]) # On résout le système linéaire pour chaque nœud de la grille en utilisant spsolve, mais on prend la valeur maximale entre la valeur de l'option et le payoff de l'option
            
                        
            elif solver == "splu":
                DD = splu(D)
                if exercise=="European":        
                    for i in range(Ntime-2,-1,-1):
                        offset[0] = a * V[0,i]
                        offset[-1] = c * V[-1,i]
                        V[1:-1,i] = DD.solve( V[1:-1,i+1] - offset )
                elif exercise=="American":
                    for i in range(Ntime-2,-1,-1):
                        offset[0] = a * V[0,i]
                        offset[-1] = c * V[-1,i]
                        V[1:-1,i] = np.maximum( DD.solve( V[1:-1,i+1] - offset ), Payoff[1:-1])
            else:
                raise ValueError("Solver is splu, spsolve, SOR or Thomas")    
            
            self.price = np.interp(x0, x, V[:,0]) # On interpole les valeurs de la matrice V pour obtenir la valeur de l'option à l'emplacement du prix initial, et on stocke le résultat dans la variable self.price
            self.price_vec = V[:,0] # On stocke la colonne correspondant au premier instant de temps de la matrice V dans la variable self.price_vec
            self.P_matrix = V
            
            if (Time == True):
                elapsed = time.time()-t_init
                return self.price, elapsed
            else:
                return self.price
  
######################### ######################### ######################### ######################### ######################### ######################### ######################### 
######################### #########################  Polynomial regression LSM (Lonstaff-Schwartz monte carlo) ######################### ######################### ######################### 
######################### ######################### ######################### ######################### ######################### #########################          

        def LSM(self, paths=10000, order=2):
            """
            Longstaff-Schwartz Method for pricing American options
        
            N = number of time steps
            paths = number of generated paths
            order = order of the polynomial for the regression 
            """
            
            start_time = time.time()  # Mesure du temps d'exécution du calcul
            
            N = self.num_steps  # Nombre de pas de temps de la simulation
            
            # Vérification de la validité du type d'option
            if self.Option_Type not in ["Call", "Put"]:
                raise ValueError("Invalid option type. Set 'Call' or 'Put'.")
            
            dt = self.T / (N - 1)  # Intervalle de temps entre chaque pas
            df = np.exp(-self.r * dt)  # Facteur d'actualisation pour chaque pas de temps
            
            #On génère les évolutions du sous jacent à l'aide de mouvmenents browniens géométriques
            X0 = np.zeros((paths, 1))  # Vecteur initial des prix de l'actif sous-jacent
            increments = ss.norm.rvs(loc=(self.r - self.sigma**2 / 2) * dt, scale=np.sqrt(dt) * self.sigma, size=(paths, N - 1))  # Incréments aléatoires pour chaque pas de temps
            X = np.concatenate((X0, increments), axis=1).cumsum(1)  # Simulation des trajectoires de prix de l'actif sous-jacent
            S = self.S0 * np.exp(X)  # Prix de l'actif sous-jacent simulé
            
            H = np.maximum(self.K - S, 0)  # Calcul des valeurs intrinsèques pour une option put
            V = np.zeros_like(H)  # Initialisation de la matrice des valeurs de l'option
            
            V[:, -1] = H[:, -1]  # Valeurs de l'option à la dernière période de décision
            
            # Boucle principale pour calculer les valeurs de l'option à chaque période de décision
            for t in range(N - 2, 0, -1):
                # Sélection des trajectoires pour lesquelles l'option a une valeur intrinsèque positive à la période t
                good_paths = H[:, t] > 0
                
                # Régression polynomiale pour estimer les valeurs de l'option à la période t + 1, on estime les coefficients optimaux par least square method
                rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, order)
                
                # Evaluation de la régression pour chaque prix simulé à la période t, on estime l'approximation de l'esperance ie : la continuation value.
                C = np.polyval(rg, S[good_paths, t])
                
                # Détermination de la décision optimale (exercer ou conserver l'option)
                exercise = np.zeros(len(good_paths), dtype=bool)
                exercise[good_paths] = H[good_paths, t] > C
                
                # Mise à jour des valeurs de l'option
                V[exercise, t] = H[exercise, t]  # Si l'option est exercée, sa valeur est égale à sa valeur intrinsèque
                V[exercise, t + 1:] = 0  # Toutes les périodes futures ont une valeur nulle si l'option est exercée
                discount_path = (V[:, t] == 0)  # Si l'option n'est pas exercée,
                V[discount_path,t] = V[discount_path,t+1]*df
        
            V0 = np.mean(V[:,1]) * df  # 
            
            return V0,time.time()-start_time
        
  
######################### ######################### ######################### ######################### ######################### ######################### ######################### 
######################### #########################  Neural network option pricing ######################### ######################### ######################### 
######################### ######################### ######################### ######################### ######################### #########################          

        

        def _build_graph(self):
            self.S = tf.placeholder(tf.float32, shape=[None, self.N+1])
            self.V = tf.Variable(tf.zeros([self.N+1, self.num_steps + 1]))
            self.Payoff = tf.maximum(self.K - self.S, 0)
    
            # Calcul du terme de diffusion
            dS = tf.gradients(self.S, [self.S])[0]
            drift = (self.r - 0.5 * self.sigma**2) * self.S
            diffusion = self.sigma * dS
            self.Mu = drift + diffusion
    
            # Définition de la grille de temps
            dt = self.T / self.num_steps
            t_grid = np.arange(self.num_steps + 1) * dt
    
            # Boucle de récurrence pour résoudre le problème
            for i in range(self.num_steps-1, -1, -1):
                dt = t_grid[i+1] - t_grid[i]
                V_new = tf.zeros([self.N+1])
                for j in range(1, self.N):
                    V_new[j] = (1 - dt * self.r) * self.V[j] + dt * tf.reduce_sum(self.Mu[j] * self.V)
                    V_new[j] = tf.maximum(V_new[j], self.Payoff[j])
                self.V.assign(V_new)
    
            # Calcul de la valeur de l'option à l'instant initial
            self.price = tf.nn.relu(tf.nn.embedding_lookup(self.V, [0])[0])

        def fit(self):
            self._build_graph()
            with tf.compat.v1.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(100):
                    S = np.random.normal(loc=self.S0, scale=self.sigma * np.sqrt(self.T), size=(10000, self.N+1))
                    price, _ = sess.run([self.price, self.V], feed_dict={self.S: S})
                    loss = np.mean(np.maximum(price - self.Payoff[:,0], 0))
                    if i % 10 == 0:
                        print(f"Iteration {i}, loss = {loss:.6f}")
                    optimizer = tf.train.AdamOptimizer()
                    train_op = optimizer.minimize(loss)
                    sess.run(train_op, feed_dict={self.S: S})

        def predict(self, S):
            with tf.compat.v1.Session() as sess:
                price = sess.run(self.price, feed_dict={self.S: S})
            return price
        
        
        def Neural_network(self):
            S = np.linspace(self.K/6, self.K*6, self.num_steps)
            S = np.tile(S, (self.N+1, 1)).T
            price = self.predict(S)
            return price
   
#### We define here some needed and usefull functions : ####

def get_stock_data(ticker, start_date, end_date):
    """
   Gets historical stock data of given tickers between dates
   :param ticker: company, or companies whose data is to fetched
   :type ticker: string or list of strings
   :param start_date: starting date for stock prices
   :type start_date: string of date "YYYY-mm-dd"
   :param end_date: end date for stock prices
   :type end_date: string of date "YYYY-mm-dd"
   :return: stock_data.csv
   """
    i = 1
    try:
        data = pdr.get_data_yahoo(ticker, start_date, end_date)
    except ValueError:
        print("Error, trying again...")
        i += 1
        if i < 5 : 
            time.sleep(10)
            get_stock_data(ticker, start_date, end_date)
        else: 
            print("Tried 5 times, Yahoo error")
            time.sleep(120)
            get_stock_data(ticker, start_date, end_date)
        stock_data = data["Adj Close"]
        stock_data.to_csv("stock_prices.csv")
    return 

def get_sp500(start_date, end_date):
    """
    Gets sp500 price data
    :param start_date: starting date for sp500 prices
    :type start_date: string of date "Y-m-d"
    :param end_date: end date for sp500 prices
    :type end_date: string of date "Y-m-d"
    :return: sp500_data.csv
    """
    i = 1
    try:
        sp500_all_data = pdr.get_data_yahoo("SPY", start_date, end_date)
    except ValueError:
        print("ValueError, trying again")
        i += 1
        if i < 5:
            time.sleep(10)
            get_stock_data(start_date, end_date)
        else:
            print("Tried 5 times, Yahoo error. Trying after 2 minutes")
            time.sleep(120)
            get_stock_data(start_date, end_date)
    sp500_data = sp500_all_data["Adj Close"]
    sp500_data.to_csv("sp500_data.csv")
    
    return 
        

        
         
def payoff(S,K,O):
    _payoff = 0
    if O == "Call":
        _payoff = max(S - K, 0)
    else :
        _payoff = max(K - S, 0)
    return _payoff


def Display_Tree(Tree):
    for i in Tree:
        print(" || ".join([str(j) for j in i]),"="*50, sep = "\n")
        
        
def main():
    
    Pricer_Call = Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = 5000, O_type = "Call")
    Pricer_Put = Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = 5000, O_type = "Put")
    
    # Test CRR Pricer :  -----------------------------------------------------------------------------------------
    Call_Price = Pricer_Call.CRR_model()
    Put_Price = Pricer_Put.CRR_model()
    print(f'\n(*) CRR_Pricer:\n\nCall Option price : {Call_Price}\nPut Option price : {Put_Price} ')
    
    '''
    runs = list(range(50,5000,50))
    CRR_Call = [Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = i, O_type = "Call").CRR_model() for i in runs]
    plt.plot(runs, CRR_Call, label='Cox_Ross_Rubinstein')
    plt.legend(loc='upper right')
    plt.show()
    
    CRR_Put = [Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = i, O_type = "Put").CRR_model() for i in runs]
    plt.plot(runs, CRR_Put, label='Cox_Ross_Rubinstein')
    plt.legend(loc='upper right')
    plt.show()'''
    
    # Test American Pricer:  -----------------------------------------------------------------------------------------
    Pricer_call = Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = 5000, O_type = "Call")
    Pricer_put = Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = 5000, O_type = "Put")
    
    Call_Price= Pricer_call.American_Price()
    Put_Price = Pricer_put.American_Price()
    print(f'\n(*) American Pricer:\n\nCall Option price : {Call_Price}\nPut Option price : {Put_Price} ')
    
    # Test American PDE :  ----------------------------------------------------------------------------------------- 
    Call_Price, Call_ExecutionTime = Pricer_call.PDE_price(Time = True,solver = "spsolve")
    Put_Price, Put_ExecutionTime = Pricer_put.PDE_price(Time = True,solver = "spsolve")
    print(f'\n(*) Partial Differential Equation (PDE) for Solver : "Spsolve":\n\nCall Option price : {Call_Price},\tExecution time : {Call_ExecutionTime}\nPut Option price : {Put_Price},\tExecution time : {Put_ExecutionTime}')
    
    Call_Price, Call_ExecutionTime = Pricer_call.PDE_price(Time = True,solver = "splu")
    Put_Price, Put_ExecutionTime = Pricer_put.PDE_price(Time = True,solver = "splu")
    print(f'\n(*) Partial Differential Equation (PDE) for Sovler : "Splu":\n\nCall Option price : {Call_Price},\tExecution time : {Call_ExecutionTime}\nPut Option price : {Put_Price},\tExecution time : {Put_ExecutionTime}')
    
    # Regression polynomiale :  -----------------------------------------------------------------------------------------
    
    Call_Price, Call_ExecutionTime = Pricer_call.LSM(order = 3)
    Put_Price, Put_ExecutionTime = Pricer_put.LSM(order = 3)
    
    print(f'\n(*) Longstaff-Schwartz Pricer (Poly regression):\n\nCall Option price : {Call_Price},\tExecution time : {Call_ExecutionTime}\nPut Option price : {Put_Price},\tExecution time : {Put_ExecutionTime}')
    
    '''
    Pricer_put = Pricer(S0 = 100, K = 110, r = 0.05, sigma = 0.3, T = 2.221918, num_steps = 5000, O_type = "Put")
    Put = Pricer_put.Neural_network()
    print(Put)'''
    
    

main()




