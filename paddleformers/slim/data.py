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
"""
Data loading utilities for quantization tasks.

This module provides functions for loading and preprocessing data from files
or default examples for model quantization.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Tuple

from utils import batchfy_text

if TYPE_CHECKING:
    from arguments import ModelArgument
    from predictor import BasePredictor


def load_calibration_data(
    model_args: "ModelArgument", predictor: "BasePredictor", batch_size: int
) -> Tuple[List, List]:
    """
    Load and preprocess data for calibration.

    This function loads data from a JSON file if specified, or uses default examples.
    It handles different data formats including single-round and multi-round conversations.

    Args:
        model_args (ModelArgument): Model arguments containing data file path.
        predictor (BasePredictor): Predictor instance with tokenizer for chat template detection.
        batch_size (int): Batch size for batching the loaded texts.

    Returns:
        Tuple[List, List]: A tuple containing:
            - batch_source_texts: Batched source texts for calibration
            - batch_target_texts: Batched target texts for calibration

    Note:
        - If `model_args.data_file` is provided, data will be loaded from the JSON file.
        - Each line in the file should be a JSON object with "instruction" and "output" fields.
        - If no data file is provided, default Chinese examples will be used.
        - The function handles both string instructions and multi-round conversation formats.
    """
    source_texts = []
    target_texts = []

    if model_args.data_file:
        with open(model_args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                # src tgt
                if isinstance(example["instruction"], str) or predictor.tokenizer.chat_template is None:
                    if isinstance(example["instruction"], str):
                        source_texts.append(example["instruction"])
                        target_texts.append(example["output"])
                    else:
                        # load multi-rounds dataset
                        source_texts.append(example["instruction"][0])
                        target_texts.append(example["output"][0])
                else:
                    source_texts.append(list(zip(example["instruction"], example["output"])))
                    target_texts.append("")
    else:
        source_texts = [
            "济南燃气结清费用需要带什么资料",
            "中通快递从辽宁到新疆要多久",
            "度小满逾期如何协商停催",
            "顺丰快递是昼夜不停的运吗?",
        ]
        target_texts = ["", "", "", ""]

    batch_source_texts = batchfy_text(source_texts, batch_size)
    batch_target_texts = batchfy_text(target_texts, batch_size)

    return batch_source_texts, batch_target_texts

