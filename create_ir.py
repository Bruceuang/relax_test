# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import tvm
from tvm import relax

import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

# Create a dummy model
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


# Give an example argument to torch.export
example_args = (torch.randn(1, 784, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(TorchModel().eval(), example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
    )

# Detach the parameters from the IRModule
mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
# Print the IRModule
mod_from_torch.show()

# Create via RelaxModule
from tvm.relax.frontend import nn


class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod_from_relax, params_from_relax = RelaxModel().export_tvm(
    {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod_from_relax.show()

######################################################################
# Create via TVMScript

from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class TVMScriptModule:
    @R.function
    def main(
        x: R.Tensor((1, 784), dtype="float32"),
        fc1_weight: R.Tensor((256, 784), dtype="float32"),
        fc1_bias: R.Tensor((256,), dtype="float32"),
        fc2_weight: R.Tensor((10, 256), dtype="float32"),
        fc2_bias: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims = R.permute_dims(fc1_weight, axes=None)
            matmul = R.matmul(x, permute_dims, out_dtype="void")
            add = R.add(matmul, fc1_bias)
            relu = R.nn.relu(add)
            permute_dims1 = R.permute_dims(fc2_weight, axes=None)
            matmul1 = R.matmul(relu, permute_dims1, out_dtype="void")
            add1 = R.add(matmul1, fc2_bias)
            gv = add1
            R.output(gv)
        return gv


mod_from_script = TVMScriptModule
mod_from_script.show()

######################################################################
# Attributes of an IRModule=

mod = mod_from_torch
print(mod.get_global_vars())

# index by global var name
print(mod["main"])
# index by global var, and checking they are the same function
(gv,) = mod.get_global_vars()
assert mod[gv] == mod["main"]

######################################################################
# Transformations on IRModules

mod = mod_from_torch
mod = relax.transform.LegalizeOps()(mod)
mod.show()

print(mod.get_global_vars())

mod = relax.get_pipeline("zero")(mod)
mod.show()

# Deploy on CPU
exec = relax.build(mod, target="llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)

raw_data = np.random.rand(1, 784).astype("float32")
data = tvm.nd.array(raw_data, dev)
cpu_out = vm["main"](data, *params_from_torch["main"]).numpy()
print(cpu_out)

# Deploy on GPU
from tvm import dlight as dl

with tvm.target.Target("cuda"):
    gpu_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)

exec = relax.build(gpu_mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)
# Need to allocate data and params on GPU device
data = tvm.nd.array(raw_data, dev)
gpu_params = [tvm.nd.array(p, dev) for p in params_from_torch["main"]]
gpu_out = vm["main"](data, *gpu_params).numpy()
print(gpu_out)

# Check the correctness of the results
assert np.allclose(cpu_out, gpu_out, atol=1e-3)