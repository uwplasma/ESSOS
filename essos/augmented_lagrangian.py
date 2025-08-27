
"""ALM (Augmented Lagrangian Method) using JAX and optimizers from OPTAX/JAXOPT/OPTIMISTIX inspired by mdmm_jax github repository"""

from typing import Any, Callable, NamedTuple
import os
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
import optax
import jaxopt
import optimistix

class LagrangeMultiplier(NamedTuple):
    """A class containing constrain parameters for Augmented Lagrangian Method"""
    value: Any
    penalty: Any
    sq_grad: Any  #For updating squared gradient in case of adaptative penalty and multiplier evolution




#This is used for the usual augmented lagrangian form 
def update_method(params,updates,eta,omega,model_mu='Constant',beta=2.0,mu_max=1.e4,alpha=0.99,gamma=1.e-2,epsilon=1.e-8,eta_tol=1.e-4,omega_tol=1.e-6):
    """Different methods for updating multipliers and penalties
    """


    pred = lambda x: isinstance(x, LagrangeMultiplier)
    if model_mu=='Constant':
        #jax.debug.print('{m}', m=model_mu)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model_mu=='Mu_Monotonic':     
        #jax.debug.print('{m}', m=model_mu)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model_mu=='Mu_Conditional_True':
        #jax.debug.print('True {m}', m=model_mu)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model_mu=='Mu_Conditional_False':
        #jax.debug.print('False {m}', m=model_mu)            
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model_mu=='Mu_Tolerance_True':
        #jax.debug.print('Standard True {m}', m=model_mu)    
        mu_average=penalty_average(params)
        #eta=eta/mu_average**(0.1)
        #omega=omega/mu_average    
        eta=jnp.maximum(eta/mu_average**(0.1),eta_tol)
        omega=jnp.maximum(omega/mu_average,omega_tol)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred),eta,omega          
    elif model_mu=='Mu_Tolerance_False':
        #jax.debug.print('Standard False {m}', m=model_mu)    
        mu_average=penalty_average(params)        
        #eta=1./mu_average**(0.1)
        #omega=1./mu_average    
        eta=jnp.maximum(1./mu_average**(0.1),eta_tol)
        #jax.debug.print('HMMMMMM mu_av {m}', m=mu_average)          
        #jax.debug.print('HMMMMMM eta {m}', m=eta)    
        omega=jnp.maximum(1./mu_average,omega_tol)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
        #return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
    elif model_mu=='Mu_Adaptative':
        #jax.debug.print('True {m}', m=model_mu)            
        #Note that y.penalty is the derivative with respect to mu and so it is 0.5*C(x)**2, like the derivative with respect to lambda is C(x)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon)*y.value,-x.penalty+gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon),-x.sq_grad+alpha*x.sq_grad+(1.-alpha)*y.penalty*2.),params,updates,is_leaf=pred)



#This is used for the squared form of the augmented Lagrangioan
def update_method_squared(params,updates,eta,omega,model_mu='Constant',beta=2.0,mu_max=1.e4,alpha=0.99,gamma=1.e-2,epsilon=1.e-8,eta_tol=1.e-4,omega_tol=1.e-6):
    """Different methods for updating multipliers and penalties)
    """


    pred = lambda x: isinstance(x, LagrangeMultiplier)
    if model_mu=='Constant':
        #jax.debug.print('{m}', m=model_mu)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier((y.value-x.value/x.penalty),0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model_mu=='Mu_Monotonic':     
        #jax.debug.print('{m}', m=model_mu)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*(y.value-x.value/x.penalty),-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model_mu=='Mu_Conditional_True':
        #jax.debug.print('True {m}', m=model_mu)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*(y.value-x.value/x.penalty),0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model_mu=='Mu_Conditional_False':
        #jax.debug.print('False {m}', m=model_mu)            
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model_mu=='Mu_Tolerance_True':
        #jax.debug.print('Squared True {m}', m=model_mu)   
        mu_average=penalty_average(params)
        #eta=eta/mu_average**(0.1)
        #omega=omega/mu_average    
        eta=jnp.maximum(eta/mu_average**(0.1),eta_tol)
        omega=jnp.maximum(omega/mu_average,omega_tol)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*(y.value-x.value/x.penalty),0.0*x.value,0.0*x.value),params,updates,is_leaf=pred),eta,omega          
    elif model_mu=='Mu_Tolerance_False':
        #jax.debug.print('Squared False {m}', m=model_mu)    
        mu_average=penalty_average(params)        
        #eta=1./mu_average**(0.1)
        #omega=1./mu_average    
        eta=jnp.maximum(1./mu_average**(0.1),eta_tol)
        omega=jnp.maximum(1./mu_average,omega_tol)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
        #return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
    elif model_mu=='Mu_Adaptative':
        #jax.debug.print('True {m}', m=model_mu)            
        #Note that y.penalty is the derivative with respect to mu and so it is 0.5*C(x)**2, like the derivative with respect to lambda is C(x)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon)*(y.value-x.value/x.penalty),-x.penalty+gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon),-x.sq_grad+alpha*x.sq_grad+(1.-alpha)*(y.penalty*2.+(x.value/x.penalty)**2)),params,updates,is_leaf=pred)




def lagrange_update(model_lagrangian='Standard'):
    """A gradient transformation for Optax that prepares an MDMM gradient
    descent ascent update from a normal gradient descent update.

    It should be used like this with a base optimizer:
        optimizer = optax.chain(
            optax.sgd(1e-3),
            mdmm_jax.optax_prepare_update(),
        )

    Returns:
        An Optax gradient transformation that converts a gradient descent update
        into a gradient descent ascent update.
    """
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(lagrange_params,updates, state,eta,omega, params=None,model_mu='Constant',beta=2.,mu_max=1.e4,alpha=0.99,gamma=1.e-2,epsilon=1.e-8,eta_tol=1.e-4,omega_tol=1.e-6):
        del params
        if model_lagrangian=='Standard' :
            return update_method(lagrange_params,updates,eta,omega,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol), state
        elif model_lagrangian=='Squared' :
            return update_method_squared(lagrange_params,updates,eta,omega,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol), state
        else:
            print('Lagrangian model not available please select Standard or Squared ')
            os._exit(0)              

    return optax.GradientTransformation(init_fn, update_fn)





class Constraint(NamedTuple):
    """A pair of pure functions implementing a constraint.

    Attributes:
        init: A pure function which, when called with an example instance of
            the arguments to the constraint functions, returns a pytree
            containing the constraint's learnable parameters.
        loss: A pure function which, when called with the the learnable
            parameters returned by init() followed by the arguments to the
            constraint functions, returns the loss value for the constraint.
    """
    init: Callable
    loss: Callable


def eq(fun,model_lagrangian='Standard', multiplier=0.0,penalty=1.,sq_grad=0., weight=1., reduction=jnp.sum):
    """Represents an equality constraint, g(x) = 0.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output zero when satisfied and smoothly
            increasingly far from zero values for increasing levels of
            constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the equality constraint.
    """

    def init_fn(*args, **kwargs):
        return {'lambda': LagrangeMultiplier(multiplier+jnp.zeros_like(fun(*args, **kwargs)),penalty+jnp.zeros_like(fun(*args, **kwargs)),sq_grad+jnp.zeros_like(fun(*args, **kwargs)))}

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs)
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    return Constraint(init_fn, loss_fn)


def ineq(fun, model_lagrangian='Standard', multiplier=0.,penalty=1., sq_grad=0.,weight=1., reduction=jnp.sum):
    """Represents an inequality constraint, h(x) >= 0, which uses a slack
    variable internally to convert it to an equality constraint.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output greater than or equal to zero when
            satisfied and smoothly increasingly negative values for increasing
            levels of constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the inequality constraint.
    """

    def init_fn(*args, **kwargs):
        out = fun(*args, **kwargs)
        return {'lambda': LagrangeMultiplier(multiplier+jnp.zeros_like(fun(*args, **kwargs)),penalty+jnp.zeros_like(fun(*args, **kwargs)),sq_grad+jnp.zeros_like(fun(*args, **kwargs))),
                'slack': jax.nn.relu(out) ** 0.5}

    if model_lagrangian=='Standard':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) - params['slack'] ** 2
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2), inf
    elif model_lagrangian=='Squared':
        def loss_fn(params, *args, **kwargs):
            inf = fun(*args, **kwargs) - params['slack'] ** 2
            return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2+ params['lambda'].value**2 /(2.*params['lambda'].penalty)), inf

    return Constraint(init_fn, loss_fn)


def combine(*args):
    """Combines multiple constraint tuples into a single constraint tuple.

    Args:
        *args: A series of constraint (init_fn, loss_fn) tuples.

    Returns:
        A single (init_fn, loss_fn) tuple that wraps the input constraints.
    """
    init_fns, loss_fns = zip(*args)

    def init_fn(*args, **kwargs):
        return tuple(fn(*args, **kwargs) for fn in init_fns)

    def loss_fn(params, *args, **kwargs):
        outs = [fn(p, *args, **kwargs) for p, fn in zip(params, loss_fns)]
        return sum(x[0] for x in outs), tuple(x[1] for x in outs)

    return Constraint(init_fn, loss_fn)



####These are auxilair functions to do operations on the lagrange multiplier parameters and on auxiliar loss information
def total_infeasibility(tree):
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.abs(y)), tree, jnp.array(0.))

#def norm_constraints(tree):
#    return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2), tree, jnp.array(0.)))

def norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.linalg.norm(flat)

def infty_norm_constraints(tree):
    flat=jax.flatten_util.ravel_pytree(tree)[0]
    return jnp.max(flat)

def penalty_average(tree):
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    penalty=jax.tree_util.tree_map(lambda x: x.penalty,tree,is_leaf=pred) 
    penalty=jax.flatten_util.ravel_pytree(penalty)        
    return jnp.average(penalty[0])








#Augmented lagrangian method classes
class ALM(NamedTuple):
    init: Callable
    update: Callable


#This can use optax gradient descent optimizers with different mu updating methods
def ALM_model_optax(optimizer: optax.GradientTransformation,  #an optimizer from OPTAX
    constraints: Constraint,     #List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Standard' ,            #Model to use for updating lagrange multipliers
    model_mu='Constant' ,            #Model to use for updating lagrange multipliers    
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):


    if model_mu=='Mu_Tolerance_LBFGS':
        @jax.jit
        def init_fn(params,**kargs):
            main_params,lagrange_params=params
            main_state = optimizer.init(main_params)
            lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)
            opt_state=main_state,lag_state
            value,grad=jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)          
            return opt_state,grad,value[0],value[1]
    else:
        @jax.jit
        def init_fn(params,**kargs):
            main_params,lagrange_params=params
            main_state = optimizer.init(main_params)
            lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)
            opt_state=main_state,lag_state
            grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)          
            return opt_state,grad,info        

    # Define the Augmented lagrangian
    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs)) #The norm here is to ensure we have a scalr from the loss which should be a vector
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        # Augmented Lagrangian
        def lagrangian_lbfgs(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm(loss(main_params,**kargs))
            mdmm_loss, _ = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss

    elif model_lagrangian=='Squared':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))   
            #Here we take the square because the term appearing in this Lagrangian
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

        # Augmented Lagrangian
        def lagrangian_lbfgs(main_params,lagrange_params,**kargs):
            #Here we take the square because the term appearing in this Lagrangian            
            main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
            mdmm_loss, _ = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss

    if model_mu=='Mu_Conditional':
        # Do the optimization step     
        @partial(jit, static_argnums=(6,7,8,9,10,11,12,13))
        def update_fn(params, opt_state,grad,info,eta,omega,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            main_updates, main_state = optimizer.update(grad[0], main_state) 
            main_params = optax.apply_updates(main_params, main_updates)
            params=main_params,lagrange_params                        
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)             
            true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Conditional_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Conditional_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            eta=norm_constraints(info[2])
            return params,opt_state,grad,info,eta,omega  
    elif model_mu=='Mu_Tolerance':
        # Do the optimization step     
        @partial(jit, static_argnums=(6,7,8,9,10,11,12,13))
        def update_fn(params, opt_state,grad,info,eta,omega,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            #While loop on omega
            state=params,main_state,grad,info
            def condition(state):
                _,_,grad,_=state
                return jnp.linalg.norm(grad[0]*main_params)> omega

            def minimization_loop(state):
                params,main_state,grad,info=state
                main_params,lagrange_params=params
                #jax.debug.print('Loop omega: {omega}', omega=omega)   
                #jax.debug.print('Loop grad: {grad}', grad=jnp.linalg.norm(grad[0]))                                              
                main_updates, main_state = optimizer.update(grad[0], main_state) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
                state=params,main_state,grad,info
                return state

            params,main_state,grad,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            eta=lag_updates[1]
            omega=lag_updates[2]
            #jax.debug.print('eta {omega}:', omega=eta)   
            #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))   
            return params,opt_state,grad,info,eta,omega   
    elif model_mu=='Mu_Tolerance_LBFGS':
        # Do the optimization step     
        @partial(jit, static_argnums=(7,8,9,10,11,12,13,14))
        def update_fn(params, opt_state,grad,value,info,eta,omega,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            #While loop on omega
            state=params,main_state,grad,value,info
            def condition(state):
                _,_,grad,_,_=state
                return jnp.linalg.norm(grad[0])> omega

            def minimization_loop(state):
                params,main_state,grad,value,info=state
                main_params,lagrange_params=params
                #jax.debug.print('Loop omega: {omega}', omega=omega)   
                #jax.debug.print('Loop grad: {grad}', grad=jnp.linalg.norm(grad[0]))                                                             
                main_updates, main_state = optimizer.update(grad[0], main_state,params=main_params,value=value,grad=grad[0],value_fn=lagrangian_lbfgs,lagrange_params=lagrange_params) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                value,grad = jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)
                #Here info is in value[1]  
                state=params,main_state,grad,value[0],value[1]
                return state

            params,main_state,grad,value,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
            params=main_params,lagrange_params
            value,grad = jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            eta=lag_updates[1]
            omega=lag_updates[2]
            #jax.debug.print('eta {omega}:', omega=eta)   
            #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))   
            return params,opt_state,grad,value[0],value[1],eta,omega                                           
    else:       
        # Do the optimization step
        @partial(jit, static_argnums=(6,7,8,9,10,11,12,13))
        def update_fn(params, opt_state,grad,info,eta,omega,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            main_updates, main_state = optimizer.update(grad[0], main_state) 
            main_params = optax.apply_updates(main_params, main_updates)
            params=main_params,lagrange_params                        
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)           
            lag_updates, lag_state = lagrange_update(model_lagrangian=model_lagrangian).update(lagrange_params,grad[1], lag_state,eta,omega,model_mu=model_mu,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            return params,opt_state, grad,info,eta,omega      


    return ALM(init_fn,partial(update_fn,model_lagrangian=model_lagrangian,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
 





#Using explicit jaxopt optimizer and not scipy wrapper, Note: JAXOPT is the only jax library with bounded lbfgs-B at the moment

#Using LBFGSB (bounded) 
def ALM_model_jaxopt_lbfgsb(constraints: Constraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Standard',
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):


    #jax.debug.print('LFBGSB {m}',m={model_lagrangian})
    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)                       
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm((loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)
    elif model_lagrangian=='Squared':
        #jax.debug.print(' LFBGSB {m}',m={model_lagrangian})
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm((loss(main_params,**kargs))))
            #This uses ||f(x)||^2 in the lagrangian
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

    @partial(jit, static_argnums=(6,7,8,9,10,11,12))
    def update_fn(params, lag_state,grad,info,eta,omega,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        minimization_loop=jaxopt.LBFGSB(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega)
        state=minimization_loop.run(main_params,bounds=(-100.*jnp.ones_like(main_params),jnp.ones_like(main_params)*100.),lagrange_params=lagrange_params,**kargs)
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))








#This uses jaxopt LevenbergMarquardt least squares (not working on jax==0.6.0 in GPU due to cuda versions conflict. Works on jax==0.5.0)
def ALM_model_jaxopt_LevenbergMarquardt(constraints: Constraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):

    model_lagrangian='Squared'


    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)                       
        return lag_state,grad,info        


    def lagrangian(main_params,lagrange_params,**kargs):
        main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        #This uses ||f(x)||^2 in the lagrangian
        return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

    #Definition to get the reisdual which for optax and optimistix least squares is going to be defined as 0.5*sum_i f_i(x)
    def lagrangian_least_residual(main_params,lagrange_params,**kargs):
        main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)    

    @partial(jit, static_argnums=(6,7,8,9,10,11,12))
    def update_fn(params, lag_state,grad,info,eta,omega,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        minimization_loop=jaxopt.LevenbergMarquardt(residual_fun=lagrangian_least_residual,has_aux=True,implicit_diff=False,xtol=1.e-14,gtol=omega)
        state=minimization_loop.run(main_params,lagrange_params=lagrange_params,**kargs)
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)




    
#This case uses JAXOPT LBFGS (unbounded version)
def ALM_model_jaxopt_lbfgs(constraints: Constraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrangian='Standard',
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):



    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)                
        return lag_state,grad,info        

    if model_lagrangian=='Standard':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.linalg.norm((loss(main_params,**kargs)))
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)
    elif model_lagrangian=='Squared':
        def lagrangian(main_params,lagrange_params,**kargs):
            main_loss = jnp.square(jnp.linalg.norm((loss(main_params,**kargs))))
            #This uses ||f(x)||^2 in the lagrangian
            mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
            return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)


    @partial(jit, static_argnums=(6,7,8,9,10,11,12))
    def update_fn(params, lag_state,grad,info,eta,omega,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        minimization_loop=jaxopt.LBFGS(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega)
        state=minimization_loop.run(main_params,lagrange_params=lagrange_params,**kargs)
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)




#####This case uses LevenbergMarquardt from optimisitix ##########
def ALM_model_optimistix_LevenbergMarquardt(constraints: Constraint,#List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):

    model_lagrangian='Squared'

    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=lagrange_update(model_lagrangian=model_lagrangian).init(lagrange_params)                
        return lag_state,grad,info        

    def lagrangian(main_params,lagrange_params,**kargs):
        main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        #This uses ||f(x)||^2 in the lagrangian
        return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

    #Definition to get the reisdual which for optax and optimistix least squares is going to be defined as 0.5*sum_i f_i(x)
    def lagrangian_least_residual(main_params,lagrange_params,**kargs):
        main_loss = jnp.square(jnp.linalg.norm(loss(main_params,**kargs)))
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  jnp.sqrt(2.*(main_loss+mdmm_loss)), (main_loss,main_loss+mdmm_loss, inf)     

    @partial(jit, static_argnums=(6,7,8,9,10,11,12))
    def update_fn(params, lag_state,grad,info,eta,omega,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        optimizer=optimistix.LevenbergMarquardt(rtol=omega,atol=omega)
        state=optimistix.least_squares(fn=lagrangian_least_residual,solver=optimizer,y0=main_params,args=lagrange_params,has_aux=True,options={'jac':'bwd'},max_steps=100000)
        main_params=state.value
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(lagrange_update(model_lagrangian=model_lagrangian).update,model_mu='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        #jax.debug.print('omega {omega}:', omega=omega)   
        #jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        #jax.debug.print('eta {omega}:', omega=eta)
        #jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))

