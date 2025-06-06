# quantization
from .quantize import (
    # not updated embedding
    SoftQuantize,
    SimpleQuantize,
    # direct updated embedding
    SoftmaxQuantize,
    ArgmaxQuantize,
    AttentionQuantize,
    # cached update embedding (Implement Later)
)

# without quantization
from .ffn import (
    LLMFFN,
    FFN,
    Identity
)
