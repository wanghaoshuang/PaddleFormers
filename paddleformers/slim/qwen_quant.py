# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import paddle
from paddle.distributed import fleet

from paddleformers.trainer.argparser import PdArgumentParser
from paddleformers.trl import llm_utils
from paddleformers.utils.log import logger

from gradual_block_quant import apply_block_gptq
from quant_utils import load_quant_model

from arguments import PredictorArgument, ModelArgument
from data import load_calibration_data
from predictor import create_predictor
from utils import benchmark

def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    llm_utils.set_triton_cache(predictor_args.model_name_or_path, predictor_args.mode)
    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    predictor = create_predictor(predictor_args, model_args)

    batch_source_texts, batch_target_texts = load_calibration_data(
        model_args, predictor, predictor_args.batch_size
    )


    if predictor_args.load_quant_path:
        load_quant_model(predictor.model, predictor_args, None, [])
    else:
        apply_block_gptq(predictor.model, predictor, batch_source_texts, batch_target_texts, predictor_args)
    if predictor_args.benchmark:
        benchmark(predictor, predictor_args, model_args)


if __name__ == "__main__":
    predict()
