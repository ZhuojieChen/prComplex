import time
import numpy as np


class Options:
    ## 类的公开属性：类的内外部均可调用，在不同实例中，类的公开属性均一致。
    # Stopping Constants
    RESIDUAL = "residual"
    ITERATIONS = "iterations"
    NORMRES = "normalizedResidual"
    RATIORES = "ratioResidual"
    HYBRIDRES = "hybridResidual"
#    ITERATIONS = "iterations"


    def __init__(self, max_iter=1e3, tolerance=1e-3, verbose=False, recordObjective=False, recordIterates=False,
                 acceleration=False, adaptive=False, **kwargs):
        #Set options parameters
        self.set(max_iter=max_iter, tolerance=tolerance, verbose=verbose, recordObjective=recordObjective,
                 recordIterates=recordIterates, accelerate=acceleration, adaptive=adaptive, **kwargs)


    def set(self, max_iter=1e3, tolerance=1e-3, verbose=False, recordObjective=False, recordIterates=False,
            accelerate=False, adaptive=False, **kwargs):   # **kwargs: 可有可无的一些参数


        STEPSIZESRHINK = "stepsizeShrink"
        FUNCTION = "function"
        STEPSIZE = "stepsize"
        RESTART = "restart"
        BACKTRACK = "backtrack"
        STOPRULE = "stopRule"
        LOOKBACK = "lookback"
        EPS_R = "EPS_R"
        EPS_N = "EPS_N"
        L = "L"

        # Required params
        self.maxIterations = max_iter
        self.verbose = verbose
        self.tolerance = tolerance
        self.recordObjective = recordObjective
        self.recordIterates = recordIterates
        self.maxIterations = max_iter  # Maximum number of iterations
        self.isAdaptive = adaptive
        self.isAccelerated = accelerate

        if STEPSIZESRHINK in kwargs:
            self.stepsizeShrink = kwargs[STEPSIZESRHINK]
        else:
            if not self.isAdaptive or self.isAccelerated:
                self.stepsizeShrink = 0.5
            else:
                self.stepsizeShrink = 0.2

        if BACKTRACK in kwargs:
            self.backtrack = kwargs[BACKTRACK]
        else:
            self.backtrack = True

        if RESTART in kwargs:
            self.restart = kwargs[RESTART]
        else:
            self.restart = True

        if STOPRULE in kwargs:
            rule = kwargs[STOPRULE]
            if rule == Options.HYBRIDRES:
                self.stopRule = rule
            elif rule == Options.RATIORES:
                self.stopRule = rule
            elif rule == Options.NORMRES:
                self.stopRule = rule
            elif rule == Options.ITERATIONS:
                self.stopRule = rule
            elif rule == Options.RESIDUAL:
                self.stopRule = rule
            else:
                raise NotImplementedError("The given stopping rule %s has not been implemented" % rule)
        else:
            self.stopRule = self.HYBRIDRES

        if EPS_R in kwargs:
            self.eps_r = kwargs[EPS_R]
        else:
            self.eps_r = 1e-8

        if EPS_N in kwargs:
            self.eps_n = kwargs[EPS_N]
        else:
            self.eps_n = 1e-8

        if LOOKBACK in kwargs:
            self.lookback = kwargs[LOOKBACK]
        else:
            self.lookback = 10

        if BACKTRACK in kwargs:
            self.backtrack = kwargs[BACKTRACK]
        else:
            self.backtrack = True

        if FUNCTION in kwargs:
            self.function = kwargs[FUNCTION]
        else:
            self.function = lambda x: 0

        self.L = None if L not in kwargs else kwargs[L]
        self.stepsize = None if STEPSIZE not in kwargs else kwargs[STEPSIZE]

    
    # stopNow 返回True时，终止迭代
    def isConverged(self, res, normRes, maxRes):
        if self.stopRule == Options.RESIDUAL:
            return res < self.tolerance
        elif self.stopRule == Options.NORMRES:
            return normRes < self.tolerance
        elif self.stopRule == Options.RATIORES:
            return res / (maxRes + self.eps_r) < self.tolerance
        elif self.stopRule == Options.HYBRIDRES:
            return (res / (maxRes + self.eps_r) < self.tolerance or normRes < self.tolerance)
        elif self.stopRule == Options.ITERATIONS:
            return False
        else:
            raise NotImplementedError("The given stopping rule %s has not been implemented and so the stop condition \
            cannot be checked" % self.stopRule)


    #Configure stepsize and Lipschitz constant specific to the current solving run
    def configure(self, x0, A, At, gradf):
        if (self.L is None or self.L <= 0) and (self.stepsize is None or self.stepsize <= 0):
            m, n = x0.shape
            x1 = np.random.randn(m, n)
            x2 = np.random.randn(m, n)
            gradf1 = At(gradf(A(x1)))
            gradf2 = At(gradf(A(x2)))

            norm_grad = np.linalg.norm(gradf1.flatten() - gradf2.flatten())
            norm_x = np.linalg.norm(x2.flatten() - x1.flatten())
            L = norm_grad / norm_x
            self.L = max(L, 1e-6)
            self.stepsize = 2 / self.L / 10

        elif self.stepsize is None or self.stepsize <= 0:
            self.stepsize = 1.0 / self.L


class Output:
    def __init__(self, solution, residuals, objectives, solveTime=None, stepsizes=None, normalizedResiduals=None,
                 functionValues=None, backtracks=None, L=None, initialStepsize=None, iterations=None):

        self.solution = solution
        self.residuals = residuals
        self.objectives = objectives
        self.solveTime = solveTime
        self.stepsizes = stepsizes
        self.normalizedResiduals = normalizedResiduals
        self.functionValues = functionValues
        self.backtracks = backtracks
        self.L = L
        self.initialStepsize = initialStepsize
        self.iterationCount = iterations

class Fasta:
    @classmethod 
    ## __类的私有方法：只能被类调用，不能在类的外部调用。
    def __isAdjoint(cls, A,At,x_in):
        x = np.random.rand(*x_in.shape)
        Ax = A(x)
        y = np.random.rand(*Ax.shape)
        Aty = At(y)
        
        ip1 = np.dot(Ax.flatten().T, y.flatten())
        ip2 = np.dot(x.flatten().T, Aty.flatten())

        error = (ip1-ip2)/max(np.abs(ip1),np.abs(ip2))
        return error < 1e-9


    def __init__(self, A, At, f, gradf, g, prox, x0, options=Options()):

        if type(A) is np.ndarray:   # 若是矩阵
            self.A = lambda x: np.dot(A,x)
        elif callable(A):   # 若是函数
            self.A = A
        else:
            raise ValueError("The input A must be a function or a matrix(type np.ndarray)")

        if type(At) is np.ndarray:
            self.At = lambda x: np.dot(At,x)
        elif callable(A):
            self.At = At
        else:
            raise ValueError("The input At must be a function or a matrix(type np.ndarray)")

#        assert Fasta.__isAdjoint(A, At, x0), "At is not the adjoint of A, please refer to the definition of these operators"

        if callable(f):
            self.f = f
        else:
            raise ValueError("f must be a callable function")

        if callable(gradf):
            self.gradf = gradf
        else:
            raise ValueError("gradf must be a callable function")

        if callable(g):
            self.g = g
        else:
            raise ValueError("g must be a callable function")

        if callable(prox):
            self.prox = prox
        else:
            raise ValueError("g must be a callable function")

        self.x0 = x0
        self.options = options

        self.options.configure(x0=x0, A=A, At=At, gradf=gradf)

    def setOptions(self, max_iter=1e3, tolerance=1e-3, verbose=False, recordObjective=False, recordIterates=False,
            accelerate=False, adaptive=False, **kwargs):
        self.options.set(max_iter, tolerance, verbose, recordObjective, recordIterates, accelerate, adaptive, kwargs)

    def run(self):
        return self.__run__(A=self.A, At=self.At, f=self.f, gradf=self.gradf, g=self.g, prox=self.prox, x0=self.x0,
                     opts=self.options)

    # __类的私有方法
    def __run__(self, A, At, f, gradf, g, prox, x0, opts):
        """""
        A - matrix or function argument that returns A*x
        At - matrix or function argument that returns A^{T} * x
        gradf - Computes the gradient of the function f at some point x 
        prox - A function argument that computes the proximal operator, a function a of z and t, prox(z,t) 
        x0 - initial iterate 
        f - a function of x which computes the value of the function f, f(x)
        g - a function of x which computes teh value of the fucntion g, g(x) 
        opts - of type options indicating the parameters for the solver 
        """""

        # get initial stepsize
        tau1 = opts.stepsize

        # maximum possible iterations
        maxIterations = int(opts.maxIterations)

        #get backtrack window from options
        lookback = opts.lookback   # window



        # Create empty arrays to hold iterate values and starting counts
        residuals = []
        normalizedResidual= []
        taus = []
        fvals = []
        objectives = []
        funcValues = []
        iterates = []

        totalBacktracks = 0
        backtrackCount = 0

        #Initialize starting iterates
        x1 = x0
        d1 = A(x1)

        gradf1 = At(gradf(d1))
        f1 = f(d1)

        fvals.append(f1)

        #If running accelerated solver, get starting acceleration iterates
        if opts.isAccelerated:
            x_accel1 = x0
            d_accel1 = d1
            alpha1 = 1

        #Markers used in monotonicity checks
        maxRes = -np.inf
        minObjectiveValue = np.inf

        for i in range(maxIterations):   # defaut 50

            startTime = time.time()   # tic

            #Rename iterates
            x0 = x1
            gradf0 = gradf1
            tau0 = tau1

            #FBS, proximal step
            x1hat = x0 - (tau0 * gradf0)
            x1 = prox(x0, x1hat, tau0)

            #Record of new values
            Dx = x1 - x0   # 前后两次迭代结果之差, 用于计算残差
            d1 = A(x1)
            f1 = f(d1)   # 保真项的值

            #Backtracking line search 回溯line search
            if opts.backtrack:
                lookback_vals = fvals[-lookback:]
                M = max(lookback_vals)

                backtrackCount = 0
                while f1 > (M + (np.real(np.dot(Dx.flatten(),gradf0.flatten()))**2)/(2*tau0)) \
                        and backtrackCount < 20:

                    #Shrink stepsize 收缩步长
                    tau0 = tau0 * opts.stepsizeShrink   

                    #FBS again 收缩步长后，重新更新本次迭代
                    x1hat = x0 - (tau0 * gradf0)
                    x1 = prox(x0, x1hat, tau0)

                    # Record of new values
                    d1 = A(x1)
                    f1 = f(d1)
                    Dx = x1 - x0   # 两次迭代结果之差

                    backtrackCount += 1

                totalBacktracks += backtrackCount

            if opts.verbose and backtrackCount >= 10:
                print("%s\nWARNING: excessive backtracking %d, current stepsize is" % (opts.header, i, tau0))

            #Record parameters and convergence information
            taus.append(tau0)

            #Gradient estimate (should be zero at solution)
            res = np.linalg.norm(Dx.flatten())/tau0

            #Record estimate
            residuals.append(res)

            #Record maximum residual
            maxRes = max(maxRes,res)

            #Compute normalizing term for residual
            normalize = max(np.linalg.norm(gradf0.flatten()),np.linalg.norm(x1.flatten()-x1hat.flatten())/tau0) + opts.eps_n

            #Compute and record normalized residual
            normRes = res/normalize
            normalizedResidual.append(normRes)

            #Record function values
            fvals.append(f1)
            funcValues.append(opts.function(x0))

            #Record objective function
            if opts.recordObjective:
                obj = f1+g(x0, x1)
                objectives .append(obj)
                newObjectiveValue = obj
            else:
                newObjectiveValue = res

            #Record iterates if needed
            if opts.recordIterates:
                iterates.append(x1)

            #Keep track of solution if applicable  跟踪解决方案；当目标值or残差很小时，得到的结果是最佳的,作为output
            if newObjectiveValue < minObjectiveValue:
                bestObjectiveIterate = x1
                minObjectiveValue = newObjectiveValue
            
            # 满足停止条件
            if opts.isConverged(res,normRes,maxRes) or i+1 >= maxIterations:
                solveTime = time.time() - startTime   # toc
                output = Output(solution=bestObjectiveIterate, residuals=residuals, objectives=objectives, solveTime=solveTime, stepsizes=taus,
                                normalizedResiduals=normalizedResidual, functionValues=funcValues,
                                backtracks=totalBacktracks, L=opts.L, initialStepsize=opts.stepsize, iterations=i)

                return output

            #FASTA Adaptive
            if opts.isAdaptive and not opts.isAccelerated:
                gradf1 = At(gradf(d1))
                Dg = gradf1 + (x1hat - x0)/tau0

                dotprod = np.real(np.dot(Dx.flatten(),Dg.flatten()))
                tau_s = (np.linalg.norm(Dx.flatten())**2)/dotprod
                tau_m = dotprod/np.linalg.norm(Dg.flatten())**2
                tau_m = max(tau_m,0)
                if 2*tau_m > tau_s:
                    tau1 = tau_m
                else:
                    tau1 = tau_s - 0.5*tau_m

                #Check stepsize is non-negative and valid
                if tau1 <= 0 or np.isinf(tau1) or np.isnan(tau1):
                    #Grow stepsize
                    tau1 = tau0*1.5

            #FISTA acceleration
            elif opts.isAccelerated:
                x_accel0 = x_accel1
                d_accel0 = d_accel1

                alpha0 = alpha1

                x_accel1 = x1
                d_accel1 = d1

                #Check acceleration for restart
                if opts.restart and np.dot((x0.flatten()-x1.flatten()).T,(x_accel1.flatten()-x_accel0.flatten())) > 0:
                    alpha0 = 1

                #Compute acceleration parameter
                alpha1 = (1+np.sqrt(1+4*(alpha0**2)))/2

                #Compute new iterate
                alpha_coeff = (alpha0-1)/alpha1
                x1 = x_accel1 + (x_accel1-x_accel0)*alpha_coeff
                d1 = d_accel1 + (d_accel1-d_accel0)*alpha_coeff

                #Calculate new gradient based on updated iterated
                gradf1 = At(gradf(d1))
                fvals.append(f(d1))
                tau1 = tau0
            
            # Plain
            else:
                gradf1 = At(gradf(d1))
                tau1 = tau0