import numpy
import scipy as sci
import scipy.optimize as sciopt


def f(coords): #(y,z) sotto forma di un narray (2,)
        return (coords[0]+3)**2+numpy.sin(coords[0])+(coords[1]+1)**2

def l_bfgs_with_approx_grad(coord):
      x,o,d=sciopt.fmin_l_bfgs_b(f,coord,approx_grad=True)
      return x

def calc_gradient(coord):#coord vettore riga
    eps=10**-17
    dim=coord.size
    gradient=[]

    for i in range(dim):
        c_eps=numpy.zeros(dim)
        c_eps[i]=eps
        gradient.append((f(coord+c_eps)-f(coord-c_eps))/(2*eps))
        print((f(coord+c_eps)-f(coord-c_eps))/(2*eps))
        #print((f(coord+c_eps)-f(coord+c_eps))/(2*eps))
    return numpy.array(gradient)

def l_bfgs_grad_given(coord):
    x,o,d=sciopt.fmin_l_bfgs_b(f(coord),coord)
    return x

if __name__=="__main__":
    coord=numpy.array([0,0])
    print(calc_gradient(coord))
    print(l_bfgs_with_approx_grad(coord))