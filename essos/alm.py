
"""ALM (Augmented Lagrangian multimplier) using JAX and OPTAX."""

from typing import Any, Callable, NamedTuple

import jax
from jax import jit
import jax.numpy as jnp
import optax
from functools import partial
import jaxopt

class LagrangeMultiplier(NamedTuple):
    """Marks the Lagrange multipliers as such in the gradient and update so
    the MDMM gradient descent ascent update can be prepared from the gradient
    descent update."""
    value: Any
    penalty: Any
    sq_grad: Any  #For updating squared gradient


def prepare_update(params,updates,eta,omega,model='Constant',beta=2.0,mu_max=1.e4,alpha=0.99,gamma=1.e-2,epsilon=1.e-8,eta_tol=1.e-4,omega_tol=1.e-6):
    """Prepares an MDMM gradient descent ascent update from a gradient descent
    update.

    Args:
        A pytree containing the original gradient descent update.

    Returns:
        A pytree containing the gradient descent ascent update.
    """
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    if model=='Constant':
        jax.debug.print('{m}', m=model)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model=='Mu_Monotonic':     
        jax.debug.print('{m}', m=model)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model=='Mu_Conditional_True':
        jax.debug.print('True {m}', m=model)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model=='Mu_Conditional_False':
        jax.debug.print('False {m}', m=model)            
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model=='Mu_Tolerance_True':
        jax.debug.print('True {m}', m=model)    
        mu_average=penalty_average(params)
        #eta=eta/mu_average**(0.1)
        #omega=omega/mu_average    
        eta=jnp.maximum(eta/mu_average**(0.1),eta_tol)
        omega=jnp.maximum(omega/mu_average,omega_tol)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred),eta,omega          
    elif model=='Mu_Tolerance_False':
        jax.debug.print('False {m}', m=model)    
        mu_average=penalty_average(params)        
        #eta=1./mu_average**(0.1)
        #omega=1./mu_average    
        eta=jnp.maximum(1./mu_average**(0.1),eta_tol)
        omega=jnp.maximum(1./mu_average,omega_tol)        
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+beta*x.penalty,0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
        #return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred),eta,omega                            
    elif model=='Mu_Adaptative':
        jax.debug.print('True {m}', m=model)            
        #Note that y.penalty is the derivative with respect to mu and so it is 0.5*C(x)**2, like the derivative with respect to lambda is C(x)
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon)*y.value,-x.penalty+gamma/(jnp.sqrt(alpha*x.sq_grad+(1.-alpha)*y.penalty*2.)+epsilon),-x.sq_grad+alpha*x.sq_grad+(1.-alpha)*y.penalty*2.),params,updates,is_leaf=pred)


def optax_prepare_update():
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

    def update_fn(lagrange_params,updates, state,eta,omega, params=None,model='Constant',beta=2.,mu_max=1.e4,alpha=0.99,gamma=1.e-2,epsilon=1.e-8,eta_tol=1.e-4,omega_tol=1.e-6):
        del params
        return prepare_update(lagrange_params,updates,eta,omega,model=model,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol), state

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


def eq(fun, multiplier=0.0,penalty=1.,sq_grad=0., weight=1., reduction=jnp.sum):
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

    def loss_fn(params, *args, **kwargs):
        inf = fun(*args, **kwargs)
        return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2), inf

    return Constraint(init_fn, loss_fn)


def ineq(fun, multiplier=0.,penalty=1., sq_grad=0.,weight=1., reduction=jnp.sum):
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

    def loss_fn(params, *args, **kwargs):
        inf = fun(*args, **kwargs) - params['slack'] ** 2
        return weight * reduction(-params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2), inf

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


class ALM(NamedTuple):
    init: Callable
    update: Callable


#Optax Gradient based transformation for Augmented Lagrange Multiplier
def ALM_model(optimizer: optax.GradientTransformation,  #an optimizer from OPTAX
    constraints: Constraint,     #List of constraints
    loss= lambda x: 0.,                    #function which represents the loss   (Callable, default 0.)
    model_lagrange='Constant' ,            #Model to use for updating lagrange multipliers
    beta=2.0,
    mu_max=1.e4,
    alpha=0.99,
    gamma=1.e-2,
    epsilon=1.e-8,
    eta_tol=1.e-4,
    omega_tol=1.e-6,
    **kargs,                   #Extra key arguments for loss
):


    


    if model_lagrange=='Mu_Tolerance_LBFGS':
        @jax.jit
        def init_fn(params,**kargs):
            main_params,lagrange_params=params
            main_state = optimizer.init(main_params)
            lag_state=optax_prepare_update().init(lagrange_params)
            opt_state=main_state,lag_state
            value,grad=jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)          
            return opt_state,grad,value[0],value[1]
    else:
        @jax.jit
        def init_fn(params,**kargs):
            main_params,lagrange_params=params
            main_state = optimizer.init(main_params)
            lag_state=optax_prepare_update().init(lagrange_params)
            opt_state=main_state,lag_state
            grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)          
            return opt_state,grad,info        

    # Augmented Lagrangian
    def lagrangian(main_params,lagrange_params,**kargs):
        main_loss = loss(main_params,**kargs)
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)

    # Augmented Lagrangian
    def lagrangian_lbfgs(main_params,lagrange_params,**kargs):
        main_loss = loss(main_params,**kargs)
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  main_loss+mdmm_loss

    if model_lagrange=='Mu_Conditional':
        # Do the optimization step     
        @partial(jit, static_argnums=(6,7,8,9,10,11))
        def update_fn(params, opt_state,grad,info,eta,omega,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            main_updates, main_state = optimizer.update(grad[0], main_state) 
            main_params = optax.apply_updates(main_params, main_updates)
            params=main_params,lagrange_params                        
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)             
            true_func=partial(optax_prepare_update().update,model='Mu_Conditional_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(optax_prepare_update().update,model='Mu_Conditional_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            eta=norm_constraints(info[2])
            return params,opt_state,grad,info,eta,omega  
    elif model_lagrange=='Mu_Tolerance':
        # Do the optimization step     
        @partial(jit, static_argnums=(6,7,8,9,10,11))
        def update_fn(params, opt_state,grad,info,eta,omega,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            #While loop on omega
            state=params,main_state,grad,info
            def condition(state):
                _,_,grad,_=state
                return jnp.linalg.norm(grad[0])> omega

            def minimization_loop(state):
                params,main_state,grad,info=state
                main_params,lagrange_params=params
                jax.debug.print('Loop omega: {omega}', omega=omega)   
                jax.debug.print('Loop grad: {grad}', grad=jnp.linalg.norm(grad[0]))                                              
                main_updates, main_state = optimizer.update(grad[0], main_state) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
                state=params,main_state,grad,info
                return state

            params,main_state,grad,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            true_func=partial(optax_prepare_update().update,model='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(optax_prepare_update().update,model='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
            params=main_params,lagrange_params
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            eta=lag_updates[1]
            omega=lag_updates[2]
            jax.debug.print('eta {omega}:', omega=eta)   
            jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))   
            return params,opt_state,grad,info,eta,omega   
    elif model_lagrange=='Mu_Tolerance_LBFGS':
        # Do the optimization step     
        @partial(jit, static_argnums=(7,8,9,10,11,12))
        def update_fn(params, opt_state,grad,value,info,eta,omega,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            #While loop on omega
            state=params,main_state,grad,value,info
            def condition(state):
                _,_,grad,_,_=state
                return jnp.linalg.norm(grad[0])> omega

            def minimization_loop(state):
                params,main_state,grad,value,info=state
                main_params,lagrange_params=params
                jax.debug.print('Loop omega: {omega}', omega=omega)   
                jax.debug.print('Loop grad: {grad}', grad=jnp.linalg.norm(grad[0]))                                                             
                main_updates, main_state = optimizer.update(grad[0], main_state,params=main_params,value=value,grad=grad[0],value_fn=lagrangian_lbfgs,lagrange_params=lagrange_params) 
                main_params = optax.apply_updates(main_params, main_updates)
                params=main_params,lagrange_params                        
                value,grad = jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)
                #Here info is in value[1]  
                state=params,main_state,grad,value[0],value[1]
                return state

            params,main_state,grad,value,info=jax.lax.while_loop(condition,minimization_loop,state)
            main_params,lagrange_params=params
            true_func=partial(optax_prepare_update().update,model='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            false_func=partial(optax_prepare_update().update,model='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
            params=main_params,lagrange_params
            value,grad = jax.value_and_grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
            opt_state=main_state,lag_state
            eta=lag_updates[1]
            omega=lag_updates[2]
            jax.debug.print('eta {omega}:', omega=eta)   
            jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))   
            return params,opt_state,grad,value[0],value[1],eta,omega                                           
    else:       
        # Do the optimization step
        @partial(jit, static_argnums=(6,7,8,9,10,11))
        def update_fn(params, opt_state,grad,info,eta,omega,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            main_updates, main_state = optimizer.update(grad[0], main_state) 
            main_params = optax.apply_updates(main_params, main_updates)
            params=main_params,lagrange_params                        
            grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)           
            lag_updates, lag_state = optax_prepare_update().update(lagrange_params,grad[1], lag_state,eta,omega,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            return params,opt_state, grad,info,eta,omega        


    return ALM(init_fn,partial(update_fn,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)



#Augmented 
def ALM_model_jaxopt_scipy(constraints: Constraint,#List of constraints
    optimizer='L-BFGS-B' ,  #the name of jax.scipy optimize  
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



    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=optax_prepare_update().init(lagrange_params)                
        return lag_state,grad,info        

    @jax.jit
    # Augmented Lagrangian
    def lagrangian(main_params,lagrange_params,**kargs):
        main_loss = loss(main_params,**kargs)
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)



  
    #@partial(jit, static_argnums=(6,7,8,9,10,11,12,13))
    def update_fn(params, lag_state,grad,info,eta,omega,optimizer=optimizer,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        minimization_loop=jaxopt.ScipyMinimize(fun=lagrangian,method=optimizer,has_aux=True,value_and_grad=False,tol=omega)
        state=minimization_loop.run(main_params,lagrange_params,**kargs)  
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(optax_prepare_update().update,model='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(optax_prepare_update().update,model='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        jax.debug.print('omega {omega}:', omega=omega)   
        jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        jax.debug.print('eta {omega}:', omega=eta)
        jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,optimizer=optimizer,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)





#Using explicit jaxopt optimizer and not scipy wrapper, Note: JAXOPT is the only jax library with bounded lbfgs at the moment
def ALM_model_jaxopt_lbfgsb(constraints: Constraint,#List of constraints
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



    @jax.jit
    def init_fn(params,**kargs):
        main_params,lagrange_params=params
        grad,info=jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        lag_state=optax_prepare_update().init(lagrange_params)                
        return lag_state,grad,info        

    @jax.jit
    # Augmented Lagrangian
    def lagrangian(main_params,lagrange_params,**kargs):
        main_loss = loss(main_params,**kargs)
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  main_loss+mdmm_loss, (main_loss,main_loss+mdmm_loss, inf)


    @partial(jit, static_argnums=(6,7,8,9,10,11,12))
    def update_fn(params, lag_state,grad,info,eta,omega,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol,**kargs):
        main_params,lagrange_params=params
        minimization_loop=jaxopt.LBFGSB(fun=lagrangian,has_aux=True,value_and_grad=False,tol=omega)
        state=minimization_loop.run(main_params,bounds=(-100.*jnp.ones_like(main_params),jnp.ones_like(main_params)*100.),lagrange_params=lagrange_params,**kargs)
        main_params=state.params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)  
        true_func=partial(optax_prepare_update().update,model='Mu_Tolerance_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)
        false_func=partial(optax_prepare_update().update,model='Mu_Tolerance_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol)            
        lag_updates, lag_state = jax.lax.cond(norm_constraints(info[2])<eta,true_func,false_func,lagrange_params,grad[1], lag_state,eta,omega)
        lagrange_params = optax.apply_updates(lagrange_params, lag_updates[0]) 
        params=main_params,lagrange_params
        grad,info = jax.grad(lagrangian,has_aux=True,argnums=(0,1))(main_params,lagrange_params,**kargs)              
        eta=lag_updates[1]
        omega=lag_updates[2]
        jax.debug.print('omega {omega}:', omega=omega)   
        jax.debug.print('grad {grad}:', grad=jnp.linalg.norm(grad[0]))           
        jax.debug.print('eta {omega}:', omega=eta)
        jax.debug.print('contraint {grad}:', grad=norm_constraints(info[2]))  
        return params,lag_state,grad,info,eta,omega                                  


    return ALM(init_fn,partial(update_fn,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,eta_tol=eta_tol,omega_tol=omega_tol))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)