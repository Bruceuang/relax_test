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

import os
import tempfile
import numpy as np
import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn

# Prepare a Relax Module
class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


input_shape = (1, 784)
mod, params = RelaxModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
mod.show()

# Library Dispatch
# Import cublas pattern
import tvm.relax.backend.contrib.cublas as _cublas


# Define a new pass for CUBLAS dispatch
@tvm.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        # Check if CUBLAS is enabled
        if not tvm.get_global_func("relax.ext.cublas", True):
            raise Exception("CUBLAS is not enabled.")

        # Get interested patterns
        patterns = [relax.backend.get_pattern("cublas.matmul_transposed_bias_relu")]
        # Note in real-world cases, we usually get all patterns
        # patterns = relax.backend.get_patterns_with_prefix("cublas")

        # Fuse ops by patterns and then run codegen
        mod = relax.transform.FuseOpsByPattern(patterns, annotate_codegen=True)(mod)
        mod = relax.transform.RunCodegen()(mod)
        return mod


mod = CublasDispatch()(mod)
mod.show()

# Auto Tuning
device = tvm.cuda(0)
target = tvm.target.Target.from_device(device)
if os.getenv("CI", "") != "true":
    trials = 2000
    with target, tempfile.TemporaryDirectory() as tmp_dir:
        mod = tvm.ir.transform.Sequential(
            [
                relax.get_pipeline("zero"),
                relax.transform.MetaScheduleTuneTIR(work_dir=tmp_dir, max_trials_global=trials),
                relax.transform.MetaScheduleApplyDatabase(work_dir=tmp_dir),
            ]
        )(mod)

    mod.show()

# DLight Rules
from tvm import dlight as dl

# Apply DLight rules
with target:
    mod = tvm.ir.transform.Sequential(
        [
            relax.get_pipeline("zero"),
            dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            ),
        ]
    )(mod)

mod.show()

# Deploy the Optimized Model
ex = relax.build(mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
data = tvm.nd.array(np.random.rand(*input_shape).astype("float32"), dev)
gpu_params = [tvm.nd.array(np.random.rand(*p.shape).astype(p.dtype), dev) for _, p in params]
gpu_out = vm["forward"](data, *gpu_params).numpy()
print(gpu_out)