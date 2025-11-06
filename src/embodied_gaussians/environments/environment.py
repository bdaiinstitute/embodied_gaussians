# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import abc
import typing
from dataclasses import dataclass

import torch


@dataclass
class EnvironmentActions: ...


@dataclass
class EnvironmentObservations: ...


class Environment(abc.ABC):
    @abc.abstractmethod
    def observe(self) -> EnvironmentObservations: ...

    @abc.abstractmethod
    def reset(self): ...

    @abc.abstractmethod
    def act(self, actions: typing.Any): ...

    @abc.abstractmethod
    def default_actions(self) -> EnvironmentActions: ...

    @abc.abstractmethod
    def step(self): ...

    @abc.abstractmethod
    def dt(self) -> float: ...

    @abc.abstractmethod
    def time(self) -> float: ...

    @abc.abstractmethod
    def num_envs(self) -> int: ...


class Task:
    @abc.abstractmethod
    def done(self) -> torch.Tensor: ...


class TaskEnvironment(Environment, Task): ...
