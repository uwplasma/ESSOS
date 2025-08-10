
"""ALM (Augmented Lagrangian multimplier) using JAX and OPTAX."""

from typing import Any, Callable, NamedTuple

import jax
from jax import jit
import jax.numpy as jnp
import optax
from functools import partial


class LagrangeMultiplier(NamedTuple):
    """Marks the Lagrange multipliers as such in the gradient and update so
    the MDMM gradient descent ascent update can be prepared from the gradient
    descent update."""
    value: Any
    penalty: Any
    sq_grad: Any  #For updating squared gradient


def prepare_update(params,updates,model='Constant',beta=2.0,mu_max=1.e4,alpha=0.999,gamma=1.e-2,epsilon=1.e-8):
    """Prepares an MDMM gradient descent ascent update from a gradient descent
    update.

    Args:
        A pytree containing the original gradient descent update.

    Returns:
        A pytree containing the gradient descent ascent update.
    """
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    if model=='Constant':
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model=='Mu_Monotonic':     
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)  
    elif model=='Mu_Conditional_True':
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(x.penalty*y.value,0.0*x.value,0.0*x.value),params,updates,is_leaf=pred)          
    elif model=='Mu_Conditional_False':
        return jax.jax.tree_util.tree_map(lambda x,y: LagrangeMultiplier(0.0*x.value,-x.penalty+jnp.minimum(beta*x.penalty,mu_max),0.0*x.value),params,updates,is_leaf=pred)        
    elif model=='Mu_Adaptative':
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

    def update_fn(lagrange_params,updates, state, params=None,model='Constant',beta=2.,mu_max=1.e4,alpha=0.999,gamma=1.e-2,epsilon=1.e-8):
        del params
        return prepare_update(lagrange_params,updates,model=model,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon), state

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
        return weight * reduction(params['lambda'].value * inf + params['lambda'].penalty* inf ** 2 / 2), inf

    return Constraint(init_fn, loss_fn)


def ineq(fun, multiplier=0.,penalty=1., weight=1., reduction=jnp.sum):
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
                'slack': jax.nn.relu(out) ** 0.5,'mu': PenaltyCoefficient(penalty+jnp.zeros_like(fun(*args, **kwargs)))}

    def loss_fn(params, *args, **kwargs):
        inf = fun(*args, **kwargs) - params['slack'] ** 2
        return weight * reduction(params['lambda'].value * inf + params['lambda'].penalty * inf ** 2 / 2), inf

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

def norm_constraints(tree):
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2), tree, jnp.array(0.))



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
    alpha=0.999,
    gamma=1.e-2,
    epsilon=1.e-8,
    tolerance = None,
    #verbose=True,
    **kargs,                   #Extra key arguments for loss
):# -> optax.GradientTransformationExtraArgs:

    #For optimizers that do not have extra arguments
    #inner = optax.with_extra_args_support(inner)

    #def init_fn(params, opt_state)
    #    main_params,mdmm
    @jax.jit
    def init_fn(params):
        main_params,lagrange_params=params
        main_state = optimizer.init(main_params)
        lag_state=optax_prepare_update().init(lagrange_params)
        opt_state=main_state,lag_state
        return opt_state

    # Define the "loss" value for the augmented Lagrangian system optimized by MDMM
    def lagrangian(params,**kargs):
        main_params, lagrange_params = params
        main_loss = loss(main_params,**kargs)
        mdmm_loss, inf = constraints.loss(lagrange_params, main_params)  
        return  main_loss+mdmm_loss, (main_loss, inf)


    if model_lagrange=='Mu_Conditional':
        # Do the optimization step     
        @partial(jit, static_argnums=(3,4,5,6,7,8))
        def update_fn(params, opt_state,eta,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            grad,info = jax.grad(lagrangian,has_aux=True)(params,**kargs)           
            opt_updates, main_state = optimizer.update(grad[0], main_state)
            true_func=partial(optax_prepare_update().update,model='Mu_Conditional_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon)
            false_func=partial(optax_prepare_update().update,model='Mu_Conditional_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[1])<eta,true_func,false_func,lagrange_params,grad[1], lag_state)
            main_params = optax.apply_updates(main_params, opt_updates)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            eta=norm_constraints(info[1])
            return params,opt_state,eta,info
    elif model_lagrange=='Mu_Conditional_Tolerance':
        # Do the optimization step    
        @partial(jit, static_argnums=(3,4,5,6,7,8))
        def update_fn(params, opt_state,eta,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            grad,info = jax.grad(lagrangian,has_aux=True)(params,**kargs)           
            opt_updates, main_state = optimizer.update(grad[0], main_state)
            true_func=partial(optax_prepare_update().update,model='Mu_Conditional_True',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon)
            false_func=partial(optax_prepare_update().update,model='Mu_Conditional_False',beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon)            
            lag_updates, lag_state = jax.lax.cond(norm_constraints(info[1])<eta,true_func,false_func,lagrange_params,grad[1], lag_state)
            main_params = optax.apply_updates(main_params, opt_updates)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            eta=norm_constraints(info[1])
            return params,opt_state,eta,info            
    else:       
        # Do the optimization step
        @partial(jit, static_argnums=(3,4,5,6,7,8))
        def update_fn(params, opt_state,eta,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon,**kargs):
            main_state,lag_state=opt_state
            main_params,lagrange_params=params
            grad,info = jax.grad(lagrangian,has_aux=True)(params,**kargs)           
            opt_updates, main_state = optimizer.update(grad[0], main_state)    
            lag_updates, lag_state = optax_prepare_update().update(lagrange_params,grad[1], lag_state,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon)
            main_params = optax.apply_updates(main_params, opt_updates)
            lagrange_params = optax.apply_updates(lagrange_params, lag_updates) 
            params=main_params,lagrange_params
            opt_state=main_state,lag_state
            return params,opt_state, eta,info        


    return ALM(init_fn,partial(update_fn,model=model_lagrange,beta=beta,mu_max=mu_max,alpha=alpha,gamma=gamma,epsilon=epsilon))
    #return optax.GradientTransformationExtraArgs(init_fn, update_fn)
