from .math_check.math_check import check_correctness_math
from .humaneval_check.humaneval_check import check_correctness_code
from .ifeval_check.ifeval_check import check_correctness_instruction_following
from .boolq_check.boolq_check import check_correctness_boolq
from .mmlu_check.mmlu_check import check_correctness_mmlu

def check_correctness(task, datum, generation):
    if task=='gsm8k':
        return check_correctness_math(datum, generation)
    elif task=='humaneval':
        return check_correctness_code(datum, generation)
    elif task=='ifeval':
        return check_correctness_instruction_following(datum, generation)
    elif task.startswith('boolq'):
        return check_correctness_boolq(datum, generation)
    # elif task.startswith('mmlu'):
    #     return check_correctness_mmlu(datum, generation)
    else:
        return check_correctness_mmlu(datum, generation)
        #raise NotImplementedError