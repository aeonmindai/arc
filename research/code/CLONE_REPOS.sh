#!/bin/bash
# Repopulate the arc-research-code/ shadow location alongside this repo.
#
# Symlinks in research/code/<domain>/<repo> point to ../../../../arc-research-code/<domain>/<repo>.
# That target lives outside the arc git repo so the 4.8 GB of code doesn't bloat git.
# Running this script re-clones every repo into the shadow location, depth=1.
#
# Usage: ./research/code/CLONE_REPOS.sh

set -e

SHADOW="$(cd "$(dirname "$0")/../../.." && pwd)/arc-research-code"
echo "Cloning into: $SHADOW"
mkdir -p "$SHADOW"

clone() {
  local url="$1" dest_subpath="$2"
  local dest="$SHADOW/$dest_subpath"
  if [ -d "$dest/.git" ] || [ -d "$dest" ]; then
    echo "  [skip] $dest_subpath (already exists)"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  git clone --depth=1 "$url" "$dest" 2>&1 | tail -1
}

echo "=== Domain 1: Weight compression ==="
clone https://github.com/Cornell-RelaxML/qtip.git              01_weight_compression/qtip
clone https://github.com/Cornell-RelaxML/quip-sharp.git        01_weight_compression/quip_sharp
clone https://github.com/Vahe1994/AQLM.git                     01_weight_compression/aqlm
clone https://github.com/IST-DASLab/gptq.git                   01_weight_compression/gptq
clone https://github.com/mit-han-lab/llm-awq.git               01_weight_compression/awq
clone https://github.com/mit-han-lab/smoothquant.git           01_weight_compression/smoothquant
clone https://github.com/mobiusml/hqq.git                      01_weight_compression/hqq
clone https://github.com/stanford-futuredata/megablocks.git    01_weight_compression/megablocks
clone https://github.com/microsoft/unilm.git                   01_weight_compression/microsoft_unilm
clone https://github.com/microsoft/BitNet.git                  01_weight_compression/bitnet
clone https://github.com/IST-DASLab/sparsegpt.git              01_weight_compression/sparsegpt
clone https://github.com/locuslab/wanda.git                    01_weight_compression/wanda
clone https://github.com/microsoft/TransformerCompression.git  01_weight_compression/slicegpt
clone https://github.com/Lucky-Lance/Expert_Sparsity.git       01_weight_compression/expert_sparsity

echo "=== Domain 2: KV compression ==="
clone https://github.com/jy-yuan/KIVI.git                      02_kv_compression/kivi
clone https://github.com/SqueezeAILab/KVQuant.git              02_kv_compression/kvquant
clone https://github.com/spcl/QuaRot.git                       02_kv_compression/quarot
clone https://github.com/facebookresearch/SpinQuant.git        02_kv_compression/spinquant
clone https://github.com/efeslab/Atom.git                      02_kv_compression/atom
clone https://github.com/FasterDecoding/SnapKV.git             02_kv_compression/snapkv
clone https://github.com/Zefan-Cai/PyramidKV.git               02_kv_compression/pyramidkv
clone https://github.com/mit-han-lab/Quest.git                 02_kv_compression/quest
clone https://github.com/snu-comparch/InfiniGen.git            02_kv_compression/infinigen

echo "=== Domain 3: Per-token speed ==="
clone https://github.com/MoonshotAI/MoBA.git                   03_per_token_speed/moba
clone https://github.com/thu-ml/SageAttention.git              03_per_token_speed/sage_attention
clone https://github.com/SafeAILab/EAGLE.git                   03_per_token_speed/eagle
clone https://github.com/FasterDecoding/Medusa.git             03_per_token_speed/medusa
clone https://github.com/Infini-AI-Lab/MagicDec.git            03_per_token_speed/magicdec
clone https://github.com/hao-ai-lab/LookaheadDecoding.git      03_per_token_speed/lookahead
clone https://github.com/mit-han-lab/duo-attention.git         03_per_token_speed/duo_attention
clone https://github.com/Dao-AILab/flash-attention.git         03_per_token_speed/flash_attention
clone https://github.com/Dao-AILab/causal-conv1d.git           03_per_token_speed/causal_conv1d
clone https://github.com/deepseek-ai/DeepSeek-V3.git           03_per_token_speed/deepseek_v3_nsa
clone https://github.com/SJTU-IPADS/PowerInfer.git             03_per_token_speed/powerinfer
clone https://github.com/microsoft/MInference.git              03_per_token_speed/minference

echo "=== Domain 4: Aggregate throughput ==="
clone https://github.com/microsoft/sarathi-serve.git           04_aggregate_throughput/sarathi_serve
clone https://github.com/microsoft/vattention.git              04_aggregate_throughput/vattention
clone https://github.com/LMCache/LMCache.git                   04_aggregate_throughput/lmcache
clone https://github.com/microsoft/DeepSpeed-MII.git           04_aggregate_throughput/deepspeed_mii
clone https://github.com/microsoft/chunk-attention.git         04_aggregate_throughput/chunk_attention
clone https://github.com/microsoft/LLMLingua.git               04_aggregate_throughput/llmlingua

echo "=== Foundation engines ==="
clone https://github.com/vllm-project/vllm.git                 06_foundation/vllm
clone https://github.com/sgl-project/sglang.git                06_foundation/sglang
clone https://github.com/flashinfer-ai/flashinfer.git          06_foundation/flashinfer
clone https://github.com/Dao-AILab/flash-attention.git         06_foundation/flash_attention_dao
clone https://github.com/deepseek-ai/FlashMLA.git              06_foundation/flashmla
clone https://github.com/NVIDIA/TensorRT-LLM.git               06_foundation/tensorrt_llm
clone https://github.com/deepspeedai/DeepSpeed.git             06_foundation/deepspeed

echo "=== Supporting (alternative architectures) ==="
clone https://github.com/ml-jku/hopfield-layers.git            99_supporting/hopfield_layers
clone https://github.com/state-spaces/mamba.git                99_supporting/mamba
clone https://github.com/BlinkDL/RWKV-LM.git                   99_supporting/rwkv
clone https://github.com/NX-AI/xlstm.git                       99_supporting/xlstm
clone https://github.com/OpenNLPLab/lightning-attention.git    99_supporting/lightning_attention

echo ""
echo "Done. Total size:"
du -sh "$SHADOW"
echo ""
echo "Symlinks in research/code/ now resolve. Try: ls research/code/01_weight_compression/qtip/qtip-kernels/src/"
