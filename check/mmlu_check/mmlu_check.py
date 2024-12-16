import re

def check_correctness_mmlu(datum, generation):
    # x = re.search(r"\\boxed{.+?([ABCD]).+?}",generation,flags=re.DOTALL)
    # if x == None:
    #     return False
    # print(x)
    # result = generation[x.span()[0]+7:x.span()[1]-1]
    # result = result.strip()
    # if '{' in result:
    #     i = result.find('{')
    #     result = result[i+1:]
    # if ['A','B','C','D'][datum['answer']] == result[0] or result==datum['choices'][datum['answer']]:
    #     return True
    # else:
    #     return False
    x = re.findall(r"\\boxed{.*?([ABCDE]).*?}",generation,flags=re.DOTALL)
    if x==[]:
        return False
    result = x[0]
    # print(result)
    if ['A','B','C','D','E'][datum['answer']] == result:
        return True
    else:
        return False
    # return True

# datum = {}
# generation = '''We will need to follow the usual rules of polynomial multiplication. We have\n\\begin{align*}\nf(x)g(x) &= (4x - 5)(2x^2 - 4x + 2)\\\\\n&= 8x^3 - 16x^2 + 8x - 10x^2 + 20x - 10 + 2x - 4\\\\\n&= 8x^3 - 26x^2 + 28x - 4\n\\end{align*}We then reduce the polynomial by taking it modulo 8. Thus, we have\n$\\boxed{\\text{(C)}}$.\n\nFinal Answer: The final answer is C.", "answer2": "I apologize for the mistake. Let's recalculate the product:\n\n\\begin{align*}\nf(x)g(x) &= (4x - 5)(2x^2 - 4x + 2)\\\\\n&= 8x^3 - 16x^2 + 8x - 10x^2 + 20x - 10 + 2x - 4\\\\\n&= 8x^3 - 26x^2 + 28x - 14\n\\end{align*}Since we are working in $\\mathbb{Z}_8[x]$, we can reduce the coefficients of the polynomial by taking them modulo 8. This gives us:\n\n$$8x^3 - 26x^2 + 28x - 14 \\equiv 2x^3 - 2x^2 + 4x + 6$$\n\nSo, $\\boxed{\\text{(B)}}$ is the correct answer.'''
# generation = '''Let's re-examine the field extension Q(√2, √3, √18) over Q.\n\nWe know that √18 = 2√3, so we can simplify the field extension: Q(√2, √3, √18) = Q(√2, √3). Now, we can use the Tower of Composition of Field Extensions:\n\nQ(√2, √3) = Q(√2) ⊗ Q Q(√3)\n\nThe degree of Q(√2) over Q is 2, since [Q(√2):Q] = 2. The degree of Q(√3) over Q is also 2, since [Q(√3):Q] = 2. By the Tower of Composition of Field Extensions, we have:\n\nQ(√2, √3) : Q] = [Q(√2) : Q][Q(√3) : Q] = 2 × 2 = 4\n\nHowever, we can further simplify this extension by observing that √2 and √3 are algebraically independent over Q. This means that the dimension of the Q-vector space spanned by 1, √2, and √3 is 3, not 4. To see why, suppose that there exist elements α, β, and γ in Q such that:\n\nα + β√2 + γ√3 = 0, where α, β, and γ are in Q.\n\nTaking conjugates of this equation, we get:\n\nα - β√2 + γ√3 = 0, α + β√2 - γ√3 = 0, and α + β√2 + γ√3 = 0\n\n Adding these equations, we obtain (4α) + (2β)√2 + (2γ)√3 = 0.\n\nDividing this equation by 2, we get 2α + β√2 + γ√3 = 0, which implies β = -2α and γ = -α.\n\nSubstituting these values back into the original equation, we get:\n\nα - 2α√2 + (-α)√3 = 0\n\nwhich simplifies to α(√2 - 1) + √3) = 0.\n\nSince α is in Q, it cannot be rational (or a mixture of rational and cubic roots), which means that √2 - 1 and √3 are Q-linearly independent. Thus, α = 0, which contradicts the assumption that β, γ are non-zero.\n\nTherefore, the Q-vector space spanned by 1, √2, and √3 has dimension 3, not 4. Hence, the degree of the field extension Q(√2, √3) over Q is:\n\n[L:Q] = [Q(√2, √3) : Q] = 3\n\nThe answer is:\n\n\\boxed{C}'''
# print(re.findall(r"(\\boxed{.*?([ABCD]).*?})",generation,flags=re.DOTALL))
# check_correctness_mmlu(datum, generation)