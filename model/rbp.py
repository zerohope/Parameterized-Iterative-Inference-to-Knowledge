import torch
from utils.model_helper import conjugate_gradient


def RBP(params_dynamic,
        state_last,
        state_2nd_last,
        grad_state_last,
        # 只在cg—rbp中使用
        update_forward_diff=None,
        eta=1.0e-5,
        truncate_iter=50,
        rbp_method='Neumann_RBP'):
  """ Three variants of Recurrent Back Propagation:
    1, RBP
    2, CG-RBP
    3, Neumann-RBP

    Context:
      Dynamic system: h(t) = f(h(t-1), w)
      Last (convergent at time step T) state: h(T)

    Args:
      params_dynamic: list, parameters w
      state_last: list, last state, i.e., h(T)
      state_2nd_last: list, 2nd last state, i.e., h(T-1)
      grad_state_last: list, gradient of loss w.r.t. last state, i.e., dl/dh(T)
      update_forward_diff: function, foward mode auto-differentiation of dynamic system
                            see detailed explanation below
      eta: float, multiples of identity used in CG to correct eigenvalues
      truncate_iter: int, truncation iteration
      rbp_method: string, specification of rbp method

    Returns:
      grad: tuple, gradient of loss w.r.t. parameters, i.e., dl/dw

    N.B.:
      1, 2nd last state h(T-1) must be detached from the computation graph
      2, update_forward_diff is only required if you use CG-RBP
          It should implement the function J(h(t), w)v, where J(h(t), w) is the Jacobian 
          of update function f(h(t), w) w.r.t. h(t) and v is the pertubation input
          You need to wrap parameters w into the update_forward_diff closure!
          The signature of update_forward_diff is:
          Args:
            input_v: list, input v
            state: list, state h(t)
          Returns:
            state_v: list, J(h(t), w)v
  """
  assert rbp_method in ['Neumann_RBP', 'CG_RBP',
                        'RBP'], "Nonsupported RBP method {}".format(rbp_method)

  # gradient of loss w.r.t. dynamic system parameters
  # print('开始rbp梯度计算')
  # import  time
  # start = time.time()
  if rbp_method == 'Neumann_RBP':
    neumann_g = None
    neumann_v = None
    neumann_g_prev = grad_state_last
    neumann_v_prev = grad_state_last

    # 对应论文 算法  line 12      U* -> state_last   U - >  state_2nd_last   neumann_v_prev-> x
    # 返回的是 line 15 中的 梯度更新值     z_star论文中的gm
    for ii in range(truncate_iter):
      #   计算状态的导数   state_last/state_2nd_last  *  grad_outputs
      neumann_v = torch.autograd.grad(
          state_last,
          state_2nd_last,
          grad_outputs=neumann_v_prev,
          retain_graph=True,
          allow_unused=True)
      # line 13
      neumann_g = [x + y for x, y in zip(neumann_g_prev, neumann_v)]
      neumann_v_prev = neumann_v
      neumann_g_prev = neumann_g

    z_star = neumann_g
  elif rbp_method == 'CG_RBP':
    # here A = I - J^T
    def _Ax_closure(x):
      JTx = torch.autograd.grad(
          state_last,
          state_2nd_last,
          grad_outputs=x,
          retain_graph=True,
          allow_unused=True)
      Ax = [m - n for m, n in zip(x, JTx)]
      JAx = update_forward_diff(Ax, state_last)
      ATAx = [m - n + eta * p for m, n, p in zip(Ax, JAx, x)]
      return ATAx

    Jb = update_forward_diff(grad_state_last, state_last)
    ATb = [m - n for m, n in zip(grad_state_last, Jb)]
    z_star = conjugate_gradient(_Ax_closure, ATb, max_iter=truncate_iter)
  elif rbp_method == 'RBP':
    z_T = [torch.zeros_like(pp).uniform_(0, 1) for pp in state_last]

    for ii in range(truncate_iter):
      z_T = torch.autograd.grad(
          state_last,
          state_2nd_last,
          grad_outputs=z_T,
          retain_graph=True,
          allow_unused=True)
      z_T = [x + y for x, y in zip(z_T, grad_state_last)]

    z_star = z_T

  # 此处的state_last,是计算出的U*,论文中的标注为u？？？？
  # end = time.time()

  # print('rbp梯度计算结束，所花时间:{}'.format(end -start))

  # 最后返回的梯度求导大部分都为0  ？   梯度求导公式?  最后一次状态的话，需要循环递进的往前推？（copy——slice）求导应该没问题。   0意味着变动不大，state的变动不大，造成的，代码里的公式是否有问题？
  # 参数123导数为0？ 更换个loss试试？只用一致性的loss？
  # 加入state last换成statefirst？  状态的改变很少，甚至为0，会不会导致导数最后为0
  # state的计算图？
  return torch.autograd.grad(
      state_last,
      params_dynamic,
      # z_star 11
      grad_outputs=z_star,
      retain_graph=False,
      allow_unused=True)
