import numpy as np

TINY=1e-18

class MyGP:
    """
    A simple implementation of the GP algorithm following Rasmussen & Williams.
    http://gaussianprocess.org/gpml/chapters/RW.pdf 
    
    Author: Rodrigo Calderón
    email: calderon@kasi.re.kr

    """
    def __init__(self, sigma_f = 1., ell_f=1., mean = 0.,fit_mean=False):
        self.ell_f = ell_f
        self.sigma_f = sigma_f
        self.mean = mean # Constant mean function for now
        self.sigma_y = TINY
        self.cov_mat = None
        self.fit_mean= fit_mean

    def __str__(self):
        return f'GP Object with $\sigma_f={self.sigma_f}$, $\ell_f={self.l}$ and mean = {self.mean}'

    def kernel(self, x1, x2, ell_f, sigma_f):
        sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
        return sigma_f**2 * np.exp(-.5 * (1/ell_f**2) * sqdist)

    def set_parameter_vector(self,theta):
        '''
        Must input tuple/list containing the hyperparameters (sigma_f,ell_f) in that order
        '''
        if self.fit_mean:
            self.sigma_f,self.ell_f,self.mean=theta
        else:
            self.sigma_f,self.ell_f=theta
    
    def update_kernel(self,theta):
        """Update the values of the hyperparameters in the kernel and recompute the matrices with these new values
        
        Args:
            theta (tuple|array): A tuple of an array containing the new values of the hyperparameters
        """
        self.sigma_f,self.ell_f=theta
        self.fit(self.train_x,self.sigma_y,cov_mat=self.cov_mat)
        
    def get_parameter_vector(self):
        if self.fit_mean:
            return np.array([self.sigma_f,self.ell_f,self.mean])
        return np.array([self.sigma_f,self.ell_f])

    def fit(self, train_x, train_sigma, cov_mat=None):
        self.train_x = train_x.reshape(-1,1)
        self.sigma_y = train_sigma
        self.cov_mat = cov_mat
        self.N = train_x.shape[0]
        if cov_mat is None:
            self.K = self.kernel(self.train_x, self.train_x, self.ell_f, self.sigma_f) + self.sigma_y**2*np.eye(self.N)
        else:
            self.K = self.kernel(self.train_x, self.train_x, self.ell_f, self.sigma_f) + (self.cov_mat.T + self.cov_mat)/2 #Ensures positive semi-definite matrix
        self.L = np.linalg.cholesky(self.K)
        self.logdet = 2*np.sum(np.log(np.diag(self.L)))

    def predict(self, train_y, test_x, return_cov=False):
        """
        Computes predictions at test points test_x and returns mean of the posterior distribution and variance at these points.
         - :: return_var = True :: is set to True by default, to get covariance set return_cov=True.
        """
        self.x_pred=test_x.reshape(-1,1)
        self.train_y = train_y
        self.N_star = len(test_x)
        self.K_star = self.kernel(self.train_x, self.x_pred, self.ell_f, self.sigma_f)
        self.K_star_star = self.kernel(self.x_pred, self.x_pred, self.ell_f, self.sigma_f)
        self.y_obs= self.train_y - self.mean
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_obs))
        self.v = np.linalg.solve(self.L, self.K_star)
        self.posterior_mu = self.mean + self.K_star.T@self.alpha
        self.posterior_sigma = self.K_star_star - self.v.T@self.v
        if return_cov:
            return self.posterior_mu, self.posterior_sigma
        return self.posterior_mu, self.posterior_sigma.diagonal()

    def sample_conditional(self,theta,n=1):
        """
        Draw n samples from the GP conditional distribution on `x_pred` coords.
        """
        self.set_parameter_vector(theta)
        self.fit(self.train_x,self.sigma_y)
        self.predict(self.train_y,self.x_pred)
        self.L_cond=np.linalg.cholesky(self.K_star_star - np.dot(self.v.T, self.v))
        return self.posterior_mu.reshape(-1,1) + np.dot(self.L_cond, np.random.normal(size=(self.N_star,n)))

    def sample_prior(self,test_x, n=1):
        """
        Sample GP on `test_x`  coords - where n is the number of samples to be drawn.
        """
        self.N_s=len(test_x)
        self.K_ss=self.kernel(test_x.reshape(-1,1),test_x.reshape(-1,1),self.l,self.sigma_f)
        self.L_prior= np.linalg.cholesky(self.K_ss + TINY*np.eye(self.N_s))
        self.f_prior = np.dot(self.L_prior, np.random.normal(size=(self.N_s,n)))
        if n==1:
            return (self.mean + self.f_prior).flatten()
        else:
            mean=self.mean*np.ones_like(test_x)
            return (mean.reshape(-1,1) + self.f_prior)

    def LML(self,theta=None):
        '''Log-Marginal Likelihood (LML) under a GP'''

        if theta is not None:
            self.update_kernel(theta)
            
        self.y_obs = self.train_y - self.mean
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_obs))
        self.yT_alpha = self.y_obs.T@self.alpha
        return -0.5*(self.N*np.log(2*np.pi) + self.logdet + self.yT_alpha)

    def optimize(self,bounds=None,method=None,update=True):
        """Optimize the value of the hyperparameters such that the LML is maximized

        Args:
            bounds (list, optional): A list with prior ranges on the hyperparameters. Defaults to None.
            method (str, optional): Method to pass to the scipy.optimize.minimize function. If method=='DE' uses differential evolution instead of minimize, which might perform better in some occasions. Defaults to None.
            update (bool, optional): Update the kernel with the best-fit values for the hyperparameters. Defaults to True.

        Returns:
            minimize: An object returned by the scipy.optimize.minimize function with likelihood values and bestfit parameters.
        """
        from scipy import optimize as opt
        nll=lambda p: -self.LML(p)
        
        if method=='DE':
            result = opt.differential_evolution(nll, bounds)
        else:
            result = opt.minimize(nll, self.get_parameter_vector(),bounds=bounds,method=method)
        
        if update:
            self.update_kernel(result.x)

        return result
    
    def plot_prediction(self, X, include_t_points=True, plot_samples=False, Nsamples=0,conditional=False,color='orange',data_c='gray',alpha=0.6,prior=False,truth=None,ax=None,data_label='Training Data',legend=False):
        """
         -- Plot mean and 2sigma CL from the GP prediction --
            One can chooose to plot samples from the prior/posterior distributions
            by setting either Conditional or Prior to true.
            Number of samples determined by Nsamples param (default 0)

        """
        if ax is None:
            try: 
                import matplotlib.pyplot as plt
                ax=plt
            except ModuleNotFoundError:
                print('Make sure that Matplotlib is installed')
        p, var = self.predict(self.train_y,X.reshape(-1,1))
        std=np.sqrt(var)
        ax.plot(X, p,c=color,lw=2,ls='-')
        if not truth is None:
            ax.plot(X,truth,label='Truth',c='k',lw=1,ls='--')
        for i in [1,2]:
            ax.fill_between(X.flatten(),
                    (p.flatten() - i*std.flatten()),
                    (p.flatten() + i*std.flatten()),
                    color=color, alpha=alpha/(1+i))
        if plot_samples:
            if conditional:
                ax.plot(X.reshape(-1,1),self.sample_conditional(Nsamples))
            elif prior:
                ax.plot(X.reshape(-1,1),self.sample_prior(X,Nsamples))
        if include_t_points:
            ax.errorbar(self.train_x, self.train_y, yerr=self.sigma_y, fmt=".",color=data_c, capsize=0,alpha=0.3,label=data_label)
            if conditional:
                ax.title(r'%.0f samples from conditional distribution'%Nsamples)
            elif prior:
                ax.title(r'%.0f samples from prior distribution'%Nsamples)
        if legend:
            ax.legend()


    def plot_likelihood(self,sigma_f,ell_f,cmap='Blues_r',norm=None,sigma_diff=True,quiet=True,ax=None,plt_bf=False,vmax=10,extend_min=False,store_path=None):
        '''
        Plot the 2-dim log marginal likehood profile as a function of (sigma_f,ell_f). Must input array of hyperparameters (sigma_f,ell_f).
        If quiet == False returns a dictionary with best-fit parameters and likehood values as well as improvement in fit.
        '''
        
        import matplotlib.colors as colors
        
        if ax is None:
            try: 
                import matplotlib.pyplot as plt
                ax=plt
            except ModuleNotFoundError:
                print('Make sure that Matplotlib is installed')
                
        results = {}
        self.X,self.Y = np.meshgrid(sigma_f,ell_f)
        self.Z = np.zeros_like(self.X)
        max_lkl=-np.inf
        for i,sf in enumerate(sigma_f):
            for j,lf in enumerate(ell_f):
                # Update the kernel with current point values of hyperparameters
                lkl = self.LML([sf,lf])
                self.Z[j,i]= lkl
                if lkl>=max_lkl:
                    max_lkl = lkl
                    results['Best-fit Parameters']=[sf,lf]
        
        #Plot the contours and color wrt lkl improvement
        CS=ax.contourf(self.X, self.Y, -self.Z, levels=np.linspace(-self.Z.flatten().max(),-self.Z.flatten().min(),50), cmap=cmap)
        
        if plt_bf:
            ax.errorbar(results['Best-fit Parameters'][0],results['Best-fit Parameters'][-1],fmt='o',c='k')
            
        if ax==plt:
            ax.xscale('log');ax.yscale('log')
            ax.xlabel(r'$\sigma_f$',fontsize='xx-large');ax.ylabel(r'$\ell_f$',fontsize='xx-large')
            cb=ax.colorbar()
            cb.set_label(label=r'$-\ln{\mathcal{L}}$',fontsize='x-large')
        else:
            ax.set_xscale('log');ax.set_yscale('log')
            ax.set_xlabel(r'$\sigma_f$',fontsize='xx-large');ax.set_ylabel(r'$\ell_f$',fontsize='xx-large')
            plt.colorbar(CS,ax=ax,label=r'$-\ln{\mathcal{L}}$')
            
        if not quiet:
            return results
    
