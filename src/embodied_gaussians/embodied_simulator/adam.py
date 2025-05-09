# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


@wp.kernel
def adam_step_kernel_vec4(
    g: wp.array(dtype=wp.vec4),  # type: ignore
    m: wp.array(dtype=wp.vec4),  # type: ignore
    v: wp.array(dtype=wp.vec4),  # type: ignore
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=wp.vec4),  # type: ignore
):
    i = wp.tid()
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g[i], g[i])
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    sqrt_vhat = wp.vec4(
        wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3])
    )
    eps_vec3 = wp.vec4(eps, eps, eps, eps)
    params[i] = params[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec3))


@wp.kernel
def adam_step_kernel_vec3(
    g: wp.array(dtype=wp.vec3),  # type: ignore
    m: wp.array(dtype=wp.vec3),  # type: ignore
    v: wp.array(dtype=wp.vec3),  # type: ignore
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=wp.vec3),  # type: ignore
):
    i = wp.tid()
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g[i], g[i])
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    sqrt_vhat = wp.vec3(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]))
    eps_vec3 = wp.vec3(eps, eps, eps)
    params[i] = params[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec3))


@wp.kernel
def adam_step_kernel_float(
    g: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=float),
):
    i = wp.tid()
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    params[i] = params[i] - lr * mhat / (wp.sqrt(vhat) + eps)


@wp.kernel
def adam_step_kernel_half(
    g: wp.array(dtype=wp.float16),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    params: wp.array(dtype=wp.float16),
):
    i = wp.tid()
    m[i] = beta1 * m[i] + (1.0 - beta1) * float(g[i])
    v[i] = beta2 * v[i] + (1.0 - beta2) * float(g[i]) * float(g[i])
    mhat = m[i] / (1.0 - wp.pow(beta1, (t + 1.0)))
    vhat = v[i] / (1.0 - wp.pow(beta2, (t + 1.0)))
    params[i] = params[i] - wp.float16(lr * mhat / (wp.sqrt(vhat) + eps))


class Adam:
    """An implementation of the Adam Optimizer
    It is designed to mimic Pytorch's version.
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    """

    def __init__(self, params=None, lrs=0.001, betas=(0.9, 0.999), eps=1e-08):
        self.m = []  # first moment
        self.v = []  # second moment
        self.set_params(params)
        if isinstance(lrs, float):
            lrs = [lrs] * len(self.params)
        self.lrs = lrs
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0

    def set_params(self, params):
        self.params = params
        if params is not None and isinstance(params, list) and len(params) > 0:
            if len(self.m) != len(params):
                self.m = [None] * len(params)  # reset first moment
            if len(self.v) != len(params):
                self.v = [None] * len(params)  # reset second moment
            for i in range(len(params)):
                param = params[i]

                if param.dtype == wp.vec3:
                    dtype = wp.vec3
                elif param.dtype == wp.vec4:
                    dtype = wp.vec4
                elif param.dtype == wp.float32:
                    dtype = wp.float32
                elif param.dtype == wp.float16:
                    dtype = (
                        wp.float32
                    )  # we always use fp32 for moments, even if params are fp16
                else:
                    raise RuntimeError(
                        f"Unsupported dtype for Warp Adam optimizer: {param.dtype}"
                    )

                if (
                    self.m[i] is None
                    or self.m[i].shape != param.shape
                    or self.m[i].dtype != param.dtype
                ):
                    self.m[i] = wp.zeros(
                        shape=param.shape, dtype=dtype, device=param.device
                    )
                if (
                    self.v[i] is None
                    or self.v[i].shape != param.shape
                    or self.v[i].dtype != param.dtype
                ):
                    self.v[i] = wp.zeros(
                        shape=param.shape, dtype=dtype, device=param.device
                    )

    def reset_internal_state(self):
        for m_i in self.m:
            m_i.zero_()
        for v_i in self.v:
            v_i.zero_()
        self.t = 0

    def step(self, grad):
        assert self.params is not None
        for i in range(len(self.params)):
            Adam.step_detail(
                grad[i],
                self.m[i],
                self.v[i],
                self.lrs[i],
                self.beta1,
                self.beta2,
                self.t,
                self.eps,
                self.params[i],
            )
        self.t = self.t + 1

    @staticmethod
    def step_detail(g, m, v, lr, beta1, beta2, t, eps, params):
        assert params.dtype == g.dtype
        assert params.shape == g.shape
        kernel_inputs = [g, m, v, lr, beta1, beta2, t, eps, params]
        if params.dtype == wp.types.float32:
            wp.launch(
                kernel=adam_step_kernel_float,
                dim=len(params),
                inputs=kernel_inputs,
                device=params.device,
            )
        elif params.dtype == wp.types.float16:
            wp.launch(
                kernel=adam_step_kernel_half,
                dim=len(params),
                inputs=kernel_inputs,
                device=params.device,
            )
        elif params.dtype == wp.types.vec3:
            wp.launch(
                kernel=adam_step_kernel_vec3,
                dim=len(params),
                inputs=kernel_inputs,
                device=params.device,
            )
        elif params.dtype == wp.types.vec4:
            wp.launch(
                kernel=adam_step_kernel_vec4,
                dim=len(params),
                inputs=kernel_inputs,
                device=params.device,
            )
        else:
            raise RuntimeError("Params data type not supported in Adam step kernels.")
