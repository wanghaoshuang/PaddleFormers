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

import sys
from abc import abstractmethod

import paddle
from paddlenlp.generation import GenerationConfig
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PretrainedModel,
    PretrainedTokenizer,
)

from arguments import PredictorArgument, ModelArgument


class BasePredictor:
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None
    ):
        if model is not None and hasattr(model, "config"):
            self.model_config = model.config
        else:
            self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer

        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = llm_utils.init_dist_env()
        self.model_config.tensor_parallel_rank, self.model_config.tensor_parallel_degree = (
            self.tensor_parallel_rank,
            self.tensor_parallel_degree,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(config.model_name_or_path)
        except:
            logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _preprocess(self, source, tgt=None):
        # if self.tokenizer.chat_template is not None:
        #     # for str -> List[str] eg. "hello"
        #     # for List[str] -> List[str]  eg. ["hello", "hello new"]
        #     # for List[List[str]] -> List[List[List[str]]]  eg. 历史对话形式,一轮
        #     #             [ [ "Hello, how are you?", "I'm doing great. How can I help you today?"],
        #     #                ["I'd like to show off how chat templating works!"], ]
        #     # for List[Dict] -> List[List[Dict]]  [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]
        #     #                                 ->  [[{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'nice'}]]
        #     if not isinstance(source, list) or not isinstance(source[0], str):
        #         source = [source]
        #     source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]
        #     if tgt is not None:
        #         source = [source[0] + tgt[0]]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
            padding=True,
            # when use chat_template, it should not add special tokens
            # chatglm2 prefix-tokens can not be tokenized into ids
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        return tokenized_source

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError

    def _postprocess(self, predictions, return_tokens=False):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if return_tokens:
            return decoded_predictions, predictions
        else:
            return decoded_predictions

    def predict(self, input_texts: str | list[str], return_tokens=False):
        tokenized_source = self._preprocess(input_texts)
        # Synchronize the HPU device for the static graph predictor
        # Ensure that configuration data read from the CPU is updated to the HPU device
        paddle.device.synchronize()
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions, return_tokens=return_tokens)
        return decoded_predictions


class DygraphPredictor(BasePredictor):
    def __init__(
        self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None, model: PretrainedModel = None, **kwargs
    ):
        super().__init__(config, tokenizer, model)
        self.model = model
        if config.dtype is not None:
            dtype = config.dtype
        else:
            raise ValueError("Please specific the model dtype.")

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                use_flash_attention=config.use_flash_attention,
                dtype=dtype,
                convert_from_hf=True,
                tensor_parallel_degree=self.tensor_parallel_degree,
                tensor_parallel_rank=self.tensor_parallel_rank,
            )
        self.model.eval()

    @paddle.no_grad()
    def _infer(self, inputs: dict[str, paddle.Tensor]):
        result = self.model.generate(
            **inputs,
            # max_new_tokens=self.config.max_length,
            # bos_token_id=self.tokenizer.bos_token_id,
            # eos_token_id=llm_utils.get_eos_token_id(self.tokenizer, self.generation_config),
            # pad_token_id=self.tokenizer.pad_token_id,
            # decode_strategy=self.config.decode_strategy,
            # temperature=self.config.temperature,
            # top_k=self.config.top_k,
            # top_p=self.config.top_p,
            # repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.7,
            repetition_penalty=1,
        )
        result = result[0]
        return result


class AutoPredictor:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def create_predictor(
        cls,
        predictor_args: PredictorArgument,
        config: PretrainedConfig,
        model_args: ModelArgument,
        tokenizer: PretrainedTokenizer = None,
        model: PretrainedModel = None,
        **kwargs,
    ):
        """
        Create a predictor

        Args:
            predictor_args (PredictorArgument): The predictor arguments.
            config (PretrainedConfig): The model configuration.
            model_args (ModelArgument): The model arguments.
            tokenizer (PretrainedTokenizer): The tokenizer.
            **kwargs: Additional keyword arguments.
        Returns:
            Predictor: The predictor.
        """
        cache_kvs_shape = None  # used for not block_attn/append_attn
        cache_k_shapes = None  # used for block_attn/append_attn
        cache_v_shapes = None  # used for block_attn/append_attn

        # static or dynamic
        execute_mode = "Dygraph" if predictor_args.mode == "dynamic" else "StaticGraph"

        # infer/ no infer
        inference_mode = ""

        predictor_class_name = execute_mode + inference_mode + "Predictor"

        import_class = sys.modules[__name__]

        # import class
        predictor_class = getattr(import_class, predictor_class_name)

        # instance
        predictor = predictor_class(
            predictor_args,
            tokenizer=tokenizer,
            model=model,
            cache_k_shapes=cache_k_shapes,
            cache_v_shapes=cache_v_shapes,
            cache_kvs_shape=cache_kvs_shape,
            model_args=model_args,
            **kwargs,
        )
        return predictor


def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    **kwargs,
):
    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    from paddlenlp.utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        predictor_args.model_name_or_path
    )

    # init chat_template for tokenizer
    llm_utils.init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)

    # # TODO(wj-Mcat): fix llama tokenzier pad_token bug

    config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()

    model = None

    # model loading
    if False: # predictor_args.inference_model: #---#
        pass #wangna
        # model = AutoInferenceModelForCausalLM.from_pretrained(
        #     predictor_args.model_name_or_path,
        #     config=config,
        #     predictor_args=predictor_args,
        #     model_args=model_args,
        #     dtype=predictor_args.dtype,
        #     tensor_parallel_degree=tensor_parallel_degree,
        #     tensor_parallel_rank=tensor_parallel_rank,
        # )
    else:
        if predictor_args.mode == "dynamic":
            # model import (gpt-3,ernie) or AutoModel
            if model_args.model_type == "gpt-3":
                sys.path.append("./gpt-3")
                from modeling import GPTForCausalLM

                model = GPTForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
                )
            elif model_args.model_type == "ernie-3.5-se":
                sys.path.append("./ernie-3.5-se")
                from modeling import Ernie35ForCausalLM

                tensor_parallel_degree = paddle.distributed.get_world_size()
                tensor_parallel_rank = paddle.distributed.get_rank()
                model = Ernie35ForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                    tensor_parallel_output=False,
                )
            else:
                with paddle.LazyGuard():
                    with paddle.no_grad():
                        model = AutoModelForCausalLM.from_pretrained(
                            predictor_args.model_name_or_path,
                            dtype=predictor_args.dtype, 
                            convert_from_hf=True, 
                            use_flash_attention=predictor_args.use_flash_attention,
                            tensor_parallel_degree=tensor_parallel_degree,
                            tensor_parallel_rank=tensor_parallel_rank,
                            tensor_parallel_output=False,
                        )
    predictor = AutoPredictor.create_predictor(predictor_args, config, model_args, tokenizer, model=model, **kwargs)

    return predictor

