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

import time
import paddle

from arguments import PredictorArgument, ModelArgument
from paddleslim.utils.log import logger
import numpy as np


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def benchmark(predictor, predictor_args: PredictorArgument, model_args: ModelArgument):
    # Just construct a simple benchmark input. We pad input to the src_length.
    test_texts = "你是百度AI，请**参考公开资料提供的信息，回答用户问题**，做到**时效性高，专业权威，客观无偏见**。\n\n### 公开资料说明\n1. 如果不同的公开资料出现矛盾且都符合正常逻辑，务必参考权威性更高的公开资料。如果根据权威性无法区分，请给用户提供多种说法。\n2. 如果用户问题对时间信息比较敏感，结合当前时间和公开资料的发布时间选择合适公开资料。\n\n### 引用说明\n1. 将相关公开资料索引用方括号包裹，置于相关内容后，例如：\"这是相关内容。[1][2]\"。\n\n### 回答要求\n1. 优先满足用户的主要需求，并且从用户问题和公开资料中挖掘用户可能的潜在需求进行满足。\n2. 优先参考公开资料中提供的信息，如果公开资料确实没有用户需求的相关信息，请你说明公开资料没有提及相关内容，并基于自身知识回答。\n3. 如果用户包含负向情绪，如焦虑/不安/困惑/气愤/孤独无助等，请你用更有人情味的风格回答。\n4. 如果用户问题涉及网站访问、平台查询、资源获取、工具使用等需求，并且公开资料中提供了对应的准确链接时,请以\"[网址名称](URL)\"的格式给出。\n\n### 背景信息\n当前时间：2025年09月12日星期五\n当前所在地：山东省济南市\n用户画像：性别: 女;年龄: 中年;手机型号: vivo XFold3;\n用户检索历史: [济南济华燃气王官庄服务站电话: 2025-09-12 16:30:28; 王官庄济华燃气营业厅: 2025-09-12 16:30:21; 燃气结清费用去哪: 2025-09-12 16:28:09; 2024年济南高中录取分数线: 2025-09-12 13:29:08; 2024年高中录取分数线: 2025-09-12 13:28:54; 2024年高中寒假放假时间表: 2025-09-12 13:28:49; 济南385分能上高中吗: 2025-09-12 13:27:40; 385分能上高中吗: 2025-09-12 13:27:26; ABS材质对人体有害吗: 2025-09-12 11:30:55; abs是什么材质?: 2025-09-12 11:30:16; 公租房小区儿童娱乐区域属于配套建设吗: 2025-09-12 09:32:17; 公租房小区没有娱乐设施吗为什么: 2025-09-12 09:30:32; 2026年取消公租房最新通知: 2025-09-12 09:30:07; 公租房小区没有娱乐设施吗: 2025-09-12 09:29:53; 公租房小区没有娱乐设施吗: 2025-09-12 09:28:02; 公租房小区没有娱乐设施合法吗: 2025-09-12 09:26:40; 小区娱乐设施谁安装: 2025-09-12 09:17:26; 监控用漏电保护器还是空气开关: 2025-09-12 08:48:27; 漏电保护器和空气开关有什么区别: 2025-09-12 08:45:56; 秋天吃什么食物最好: 2025-09-12 08:25:17; 红花如意丸的功效与作用: 2025-09-11 10:57:15; 雪莲果是凉性的还是热性的: 2025-09-11 09:09:30; 狗狗不吃饭但精神很好: 2025-09-10 20:26:16; 狗狗不吃饭是怎么回事: 2025-09-10 20:25:04; 狗用脚挠痒痒怎么回事: 2025-09-10 20:24:23; 百香果的功效和作用: 2025-09-10 13:02:05; 劳务合同属于什么合同类型: 2025-09-09 16:28:58; 劳务合同属于行政合同吗: 2025-09-09 16:28:35; 自来水地埋管是什么材料做的: 2025-09-09 14:38:29; 2025年1月灭火器更换新规定: 2025-09-09 09:27:37; 新灭火器第一次换粉是什么时候: 2025-09-09 09:26:43; 口臭是胃火还是肝火: 2025-09-08 19:42:29; ]\n\n### 公开资料\n[1] 标题: 奥德集团有限公司费县分公司办事服务 \n参考特征:内容权威性非常高， 作者权威性非常高， 时效性较高， 发布于2025-05-27\n正文: (一)报装资料、申请受理: 1.开发商及村委集体用户安装:单体楼的建设工程规划许可证复印件、单体楼建筑施工图、小区整体平面图电子版一份。 2.零散居民户安装:即前期社区整体安装燃气时未安装的用户,出示房产证或者购房合同原件。 3.非居民用户:工商用户及小微用户等非居民用户燃气报装资料需提供用气地址的产权资料、营业执照、有效身份证明。\n\n\n[2] 标题: 燃气缴费、维修及相关服务办理程序、线上线下办理渠道、时限、网点设置、服务标准、服务承诺和便民措施 \n参考特征:内容权威性较高， 作者权威性非常高， 时效性较高， 发布于2025-02-19\n正文: 1、用户充值流程: (1)营业厅充值流程: 用户携带燃气卡、本到营业网点柜台→递交燃气卡、本→用户付款→营业员核对信息→系统充值→打印收款收据→递还燃气卡、本、收据→充值完成。 (2)政务大厅充值流程: 用户携带身份证、燃气卡、本到政务大厅→自助叫号机叫号→柜台钱等候叫号→递交燃气卡、本→用户付款→营业员核对信息→系统充值→打印收款收据→递还燃气卡、本、收据→充值完成。 (3)线上充值流程: 微信关注"奥德悦生活"微信公众号→首次登陆绑定用户编号→选择购气量→在线支付→到就近自助写卡机(实时更新,可在公众号内查询)写卡。\n\n\n[3] 标题: 燃气民用销户办事指南 \n参考特征:内容权威性非常高， 作者权威性非常高， 时效性较高， 发布于2025-06-06\n正文: 用户携带户主身份证原件及复印件、天然气用户卡、银行卡复印件(退还预存燃气费),并提交销户申请。\n\n\n[4] 标题: 【办事服务】2025年山东长乐集团民生燃气有限公司用气申请、过户、销户等项目办事服务指南\n参考特征:内容权威性较高， 作者权威性非常高， 时效性较高， 发布于2025-06-03\n正文: 一、申请 单位或是小区统一安装由单位、小区物业办公室或开发公司向燃气公司提出安装申请,填写申请单,并提供小区平面图;散户安装(1)持有效身份证件及现金(如开发商代收,需带天然气配套设施收费票据或证明)(2)签订居民燃气供用气合同。\n\n### 用户问题\n济南燃气结清费用需要带什么资料", 
    
    benchmark_texts = [
        test_texts
    ]

    batch_benchmark_texts = batchfy_text(benchmark_texts, 1)
    print("***********Start Benchmark**********")

    warmup_time = 5
    test_time = 20

    print("***********Start Warmup**********")
    for _ in range(warmup_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            predictor.predict(batch_source_text)

    print("***********Start Speed Test**********")
    start = time.perf_counter()
    output_tokens = 0
    for _ in range(test_time):
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            results = predictor.predict(batch_source_text, return_tokens=True)
            if predictor.tensor_parallel_rank == 0:
                output_tokens += sum([len(tokens) for tokens in results[-1]])
    end = time.perf_counter()
    if predictor.tensor_parallel_rank == 0:
        print("Avg Elapse time is: ", (end - start) / test_time)
        print("Output tokens is: ", output_tokens)
        print(
            "Input length is: {}, Output length is: {}, bs is: {}, IPS: {:.3f} tokens/s, QPS: {:.3f} requests/s. ".format(
                16384,
                1024,
                1,
                (output_tokens / (end - start)),
                (1 * test_time / (end - start)),
            )
        )

def load_sharded_checkpoint(folder, variant=None, return_numpy=False):
    """

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.

    """
    # Load the index
    pdparams_file = os.path.join(folder, _add_variant("model_state.pdparams", variant))
    lora_pdparams_file = os.path.join(folder, _add_variant("lora_model_state.pdparams", variant))
    safetensors_file = os.path.join(folder, _add_variant("model.safetensors", variant))
    if os.path.isfile(pdparams_file):
        return paddle.load(pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(lora_pdparams_file):
        return paddle.load(lora_pdparams_file, return_numpy=return_numpy)
    if os.path.isfile(safetensors_file):
        try:
            from paddlenlp.utils.safetensors import fast_load_file as safe_load_file
        except:
            from safetensors.numpy import load_file as safe_load_file

        state_dict = safe_load_file(safetensors_file)
        if not return_numpy:
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], np.ndarray):
                    state_dict[key] = paddle.Tensor(state_dict.pop(key), zero_copy=True)
        return state_dict

    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
    safe_master_file = os.path.join(folder, _add_variant(SAFE_MASTER_WEIGHTS_INDEX_NAME, variant))
    safe_peft_file = os.path.join(folder, _add_variant(SAFE_PEFT_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    safe_master_present = os.path.isfile(safe_master_file)
    safe_peft_present = os.path.isfile(safe_peft_file)

    load_safe = False
    load_index = None
    if safe_index_present:
        load_safe = True  # load safe due to preference
        load_index = safe_index_file
    elif safe_master_present:
        load_safe = True
        load_index = safe_master_file
    elif index_present:
        load_index = index_file
    elif safe_peft_present:
        load_safe = True
        load_index = safe_peft_file
    else:
        raise ValueError(f"Could not find {index_file} or {safe_index_file} or {safe_peft_file}")

    if load_safe:
        try:
            from paddlenlp.utils.safetensors import fast_load_file as safe_load_file
        except:
            from safetensors.numpy import load_file as safe_load_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    loader = safe_load_file if load_safe else partial(paddlenlp_load, map_location="np" if return_numpy else "cpu")

    ret = {}
    for shard_file in tqdm(shard_files):
        state_dict = loader(os.path.join(folder, shard_file))
        ret.update(state_dict)

    if not return_numpy:
        for key in list(ret.keys()):
            if isinstance(ret[key], np.ndarray):
                ret[key] = paddle.Tensor(ret.pop(key), zero_copy=True)

    return ret


def show_progress(start, idx, steps):
    """
    Show progress
    """
    c = idx / steps * 100
    a = "*" * int(c)
    b = "·" * (100 - int(c))
    dur = time.perf_counter() - start
    logger.info("\r{:.2f}%[{}->{}] Cost time {:.2f}s".format(c, a, b, dur))
    time.sleep(0.1)

def offload_params(sub_layer: paddle.nn.Layer, sub_name: str) -> dict:
    """
    Get parameters in sublayer and return as a dictionary of numpy arrays in host memory.
    """
    state_dict = {}
    for name, param in sub_layer.named_parameters():
        full_name = sub_name + "." + name
        state_dict[full_name] = np.array(param.value().get_tensor())
    return state_dict


@paddle.no_grad()
def _clear_params(model, state_dict=None, sub_name=None):
    """
    Clear params
    """
    for k, v in model.state_dict().items():
        # 清除参数的值
        v.value().get_tensor()._clear()
        # if state_dict is not None:
        #     拼接参数名
        #    name = sub_name + "." + k
        #    if name in state_dict:
        #     如果拼接后的参数名在state_dict中存在
        #    if name in state_dict:
        #            从state_dict中删除该参数
        #        del state_dict[sub_name + "." + k]


def load_params_from_cpu(sub_layer: paddle.nn.Layer, sub_name: str, state_dict: dict, dtype: str):
    """
    Load the parameters of sublayer from cpu memory to gpu memory.
    Args:
        sub_layer: The sublayer to load the parameters.
        sub_name: The name of the sublayer.
        state_dict: The state dict to load the parameters from. The keys are the names of the parameters. The values are the numpy arrays in host memory.
        dtype: The dtype of the parameters.
    """
    new_dict = {}
    for k, v in state_dict.items():
        if sub_name in k:
            weight_name = k.replace(sub_name + ".", "")
            # load from numpy, so we need to convert to bfloat16 firstly and then cast to other dtype
            new_dict[weight_name] = paddle.to_tensor(v, dtype='bfloat16').cast(dtype).cuda()
    for k, v in sub_layer.state_dict().items():
        if not v._is_initialized():
            v.get_tensor()._share_data_with(new_dict[k].get_tensor())
    sub_layer.set_state_dict(new_dict)