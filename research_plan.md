# Why Transformers Fail at Arithmetic
### A Research Problem Statement and Self-Study Curriculum
*Mathematical and Theoretical Research Program*

---

# Part I — The Problem Statement

## 1.1 Background and Motivation

Large language models based on the transformer architecture have demonstrated remarkable capabilities across a wide range of tasks involving natural language. Yet one of the most elementary computational tasks — multi-digit arithmetic — remains a persistent and systematic point of failure. A model trained on billions of tokens, capable of nuanced legal reasoning or literary analysis, reliably breaks down when asked to add two 12-digit numbers it has not seen in training.

This failure is not accidental and not primarily a matter of insufficient training data. It reflects something deep about the computational character of the transformer architecture and its relationship to the structure of arithmetic as a formal computation. Understanding this gap rigorously requires tools from circuit complexity theory, formal language theory, and statistical learning theory — a combination rarely assembled in the machine learning literature.

## 1.2 The Core Phenomenon

The problem manifests in two distinct but related ways:

- **In-distribution failure:** Transformers trained on n-digit arithmetic with fixed n learn statistical correlations that approximate the correct answer but do not implement the underlying algorithm. Accuracy degrades with digit count even within the training distribution when n is large.

- **Length generalization failure:** A transformer trained on arithmetic up to n digits fails catastrophically on (n+1)-digit arithmetic — a task that should require no new algorithmic knowledge, only application of the same procedure to a longer input.

---

> **Central Research Question**
>
> What is the precise computational-theoretic characterization of why transformer architectures — as formal mathematical objects — cannot implement length-invariant arithmetic algorithms, and why gradient-based learning fails to discover correct algorithms even when they fall within the model's expressible function class?

---

## 1.3 Why This Question Is Hard

The difficulty of this problem is not in its statement. It is in the gap between two levels of explanation that currently exist in the literature:

- **Expressibility results (what exists):** Several results establish that transformers with hard attention compute functions in the complexity class AC⁰, and that arithmetic over arbitrary-length inputs requires at minimum TC⁰ — a provably larger class. This is a clean theoretical story.

- **Learning dynamics (what is found):** The expressibility story does not explain the empirical picture. Even on tasks where correct algorithms are demonstrably within the model's expressible function class (e.g., modular arithmetic on small moduli), gradient descent frequently fails to find them, instead converging to length-specific heuristics. The phenomenon of grokking — where correct algorithms emerge only after extended training with heavy regularization — suggests the failure is as much a property of the loss landscape as of the architecture.

The open frontier lies between these two levels: a principled account of why the optimization process has an inductive bias against length-generalizing solutions, and what formal properties such solutions would need to have to be discoverable by gradient descent.

## 1.4 Formal Problem Decomposition

The research problem decomposes into at least three distinct sub-questions, each of which is partially open.

### Sub-Problem 1 — Expressibility

What is the tightest formal characterization of the function class computed by a transformer of fixed depth and width, as a function of its positional encoding scheme? Existing results (Hahn 2020, Merrill & Sabharwal 2023) establish AC⁰ bounds for hard attention. The soft attention case, which is the empirically relevant case, is significantly less understood. The question is whether there exists a positional encoding scheme under which a fixed-depth transformer can implement a function that is not length-bounded — and if so, what algebraic properties that encoding must have.

### Sub-Problem 2 — Positional Encoding and Length Invariance

Arithmetic is a length-invariant computation: the algorithm for adding two 15-digit numbers is the same as the algorithm for adding two 5-digit numbers, differing only in the number of times it is applied. What formal property of a positional encoding scheme is necessary and sufficient for a transformer to represent a length-invariant function? No existing PE scheme (sinusoidal, learned absolute, RoPE, ALiBi) provably satisfies this condition for arithmetic. Characterizing the required property is a concrete open problem.

### Sub-Problem 3 — Learning Dynamics and Inductive Bias

Even granting that a length-generalizing algorithm is expressible in principle, why does stochastic gradient descent on finite training sets consistently fail to find it? The training distribution on n-digit numbers is consistent with infinitely many functions, some that generalize and most that do not. What characterizes the set of functions that gradient descent converges to? What regularization conditions, if any, suffice to bias the search toward the generalizing solution? The grokking literature provides partial empirical answers, but a theoretical account grounded in loss landscape geometry and implicit regularization is missing.

## 1.5 Known Results and Where the Frontier Is

| Result | Status | Key Reference |
|---|---|---|
| Hard-attention transformers compute AC⁰ functions | Established | Hahn (2020) |
| Integer addition is in TC⁰ | Classical result | NC/TC complexity theory |
| PARITY ∉ AC⁰ | Established (Håstad 1987) | Furst-Saxe-Sipser, Håstad |
| Saturated transformers = constant-depth threshold circuits | Established | Merrill & Sabharwal (2023) |
| No standard PE scheme enables length generalization for arithmetic | Empirically established, not formally proved | Anil et al. (2022), Zhou et al. (2024) |
| Formal property PE must have for length invariance | **OPEN** | — |
| Why SGD fails to find generalizing algorithms | **OPEN** | Grokking literature (partial) |
| Tight soft-attention complexity characterization | **OPEN** (partially) | Merrill et al. (ongoing) |

---

# Part II — The Research Curriculum

---

> **Governing Principle**
>
> Study everything with one question as your north star: *what can be computed in parallel versus what requires sequential depth?* Every subject in this curriculum illuminates a different facet of that question. Keeping it in view makes the subjects feel unified rather than disconnected.

---

The curriculum is organized into six sequential stages spanning approximately 18–24 months of serious part-time study (or 12–18 months of full immersion). The stages are not fully sequential — Stages 1a, 1b, and 1c run in parallel, and empirical literature reading should begin at Stage 5 and continue concurrently. The critical constraint is depth, not breadth: Stage 2 in particular must be internalized at the proof level before the research becomes tractable.

---

## Stage 0 — Mathematical Maturity
*Weeks 1–8*

Before any subject-matter content, you need fluency in the language of mathematics itself. Without this foundation, all downstream learning is slower by an order of magnitude and the research is inaccessible.

**Primary Text**
- Velleman, D.J. — *How to Prove It: A Structured Approach* (3rd ed.) — Complete every exercise. The goal is not the content but the muscle memory of constructing and reading proofs.

**Supplement**
- Hammack, R. — *Book of Proof* (free online) — Second perspective on induction, set theory, and combinatorics.

**What You Are Building**

The ability to read a theorem statement, immediately parse its logical structure, and either prove it or construct a counterexample. You should finish Stage 0 treating formal symbols as a native language. Without this, you will read Sipser and Arora-Barak slowly and painfully.

---

## Stage 1 — The Three Mathematical Pillars
*Months 2–8*

These three run partially in parallel and form the true mathematical foundation of the research. They can be interleaved — for example, rotating between linear algebra, analysis, and algebra on a weekly schedule.

### 1a. Linear Algebra

- Axler, S. — *Linear Algebra Done Right* (3rd ed.) — Not a computational course. Axler builds from vector space axioms up, proving everything. This is the right approach because the research requires reasoning about linear maps abstractly, not numerically.

Internalize deeply: eigenvalues and eigenvectors, inner product spaces, the spectral theorem, dual spaces. The softmax operation in transformers is a map on a high-dimensional inner product space; understanding it abstractly requires this foundation.

### 1b. Real Analysis

- Abbott, S. — *Understanding Analysis* (2nd ed.) — The correct pedagogical entry point. Abbott teaches why the machinery exists, not just the machinery.
- Folland, G.B. — *Real Analysis* (selected chapters) — After Abbott, skim the measure theory and Lp space chapters.

Internalize deeply: sequences and their limits, continuity as a local-in-epsilon property, compactness, uniform convergence and its distinction from pointwise convergence.

### 1c. Abstract Algebra

- Dummit, D.S. & Foote, R.M. — *Abstract Algebra* (3rd ed.) — selected chapters only

You do not need all of Dummit & Foote. What you need, deeply:

- **Groups** — completely and rigorously: subgroups, cosets, homomorphisms, quotient groups, the three isomorphism theorems, group actions.
- **Semigroups and monoids** — underemphasized in standard algebra courses but central to formal language theory. The syntactic monoid of a formal language determines what automaton can recognize it.
- **Rings and fields** — lightly, for comfort with the notation.

---

> **Why Semigroup Theory Is Not Optional**
>
> The algebraic structure of carry propagation in integer addition is a specific kind of semigroup. Whether a finite automaton can compute a function is determined entirely by whether its transition monoid has a property called *aperiodicity*. This is not background material — it is the theory. The connection runs: **group theory → semigroup theory → syntactic monoids → star-free languages → aperiodic automata → what transformers can compute.** Most researchers never see this thread because they studied algebra and TCS in separate courses.

---

## Stage 2 — Theoretical Computer Science Core
*Months 6–14*

This is the heart of the curriculum. If Stage 1 is the language, this is the subject matter. Stage 2 must be internalized at the proof level before the research is tractable. The single most important technical result in this curriculum is the Håstad switching lemma — spend whatever time is necessary to understand the proof, not just the statement.

### 2a. Theory of Computation

- Sipser, M. — *Introduction to the Theory of Computation* (3rd ed.) — Read it twice: once quickly for overview, once slowly with every exercise completed.

Internalize completely: finite automata and regular languages, the pumping lemma as an adversarial lower bound argument, context-free languages and pushdown automata, Turing machines and decidability, the definitions of P and NP from first principles.

The crucial insight Sipser teaches: the pumping lemmas are not tricks. They are the first instances of the pattern "for any machine of this type, I can construct an input that forces it to fail." This adversarial universality is the same intellectual structure used throughout circuit complexity lower bounds.

### 2b. Computational Complexity

- Arora, S. & Barak, B. — *Computational Complexity: A Modern Approach* — Chapters 1–6 and Chapter 14 (circuit complexity) are essential. The rest is useful context.

Internalize completely:
- **Circuit complexity and circuit families** — the formal definition, uniformity conditions, and closure properties.
- **AC⁰ and TC⁰** — their definitions and known upper bounds. Integer addition is in TC⁰; PARITY is not in AC⁰.
- **The Håstad switching lemma** — the technical engine behind the Furst-Saxe-Sipser theorem. Understand the proof at the level of being able to reconstruct it. This is the hardest material in the curriculum.
- **Randomized complexity (BPP)** — lightly, for context.

### 2c. Formal Language Theory

- Hopcroft, J., Motwani, R., & Ullman, J. — *Introduction to Automata Theory, Languages, and Computation* (3rd ed.) — Deepens Sipser with algebraic machinery.

Additional material beyond Sipser that you specifically need:
- **String transducers** — a transducer is a finite automaton that produces output rather than accepting/rejecting. Multi-digit arithmetic is a transduction problem, not a recognition problem. This distinction is foundational.
- **The Myhill-Nerode theorem** — the right algebraic characterization of regular languages, connecting directly to semigroup theory.
- **Star-free languages and aperiodic monoids** — a language is star-free if and only if its syntactic monoid is aperiodic (Schützenberger 1965). Transformers with hard attention recognize exactly star-free languages. This is the algebraic ceiling.

### 2d. Communication Complexity

- Kushilevitz, E. & Nisan, N. — *Communication Complexity* — Shorter and more focused than the other Stage 2 texts. Read Chapters 1–4 thoroughly.

The model: Alice holds part of the input, Bob holds the rest, and they want to compute a function with minimal communication. In a transformer, position *i* and position *j* communicate only through the attention mechanism. The communication complexity of computing carry across a digit sequence lower-bounds the number of attention layers needed. This is one of the cleanest formal arguments available for Sub-Problem 1.

---

## Stage 3 — Information Theory and Probability
*Months 10–14*

Stage 3 runs partially in parallel with Stage 2. The information-theoretic material reinforces the communication complexity perspective.

### 3a. Information Theory

- Cover, T.M. & Thomas, J.A. — *Elements of Information Theory* (2nd ed.) — Chapters 1–8: entropy, mutual information, channel capacity, the data processing inequality.

The data processing inequality is a formal version of the intuition that transformations can only lose information. Arguments about what a transformer's intermediate representations cannot encode about position are ultimately information-theoretic statements of this type.

### 3b. Probability Theory

- Durrett, R. — *Probability: Theory and Examples* (5th ed.) — Chapters 1–3 at minimum: measure-theoretic foundations, laws of large numbers, the central limit theorem.

This level of probability theory is required for statistical learning theory. The PAC learning framework is built on measure-theoretic probability, and you will not be able to read it rigorously without this foundation.

---

## Stage 4 — Statistical Learning Theory
*Months 14–18*

- Shalev-Shwartz, S. & Ben-David, S. — *Understanding Machine Learning: From Theory to Algorithms* — The right level for this research. Rigorous without being inaccessible.

Internalize deeply:
- The formal definition of a hypothesis class and its complexity measures (VC dimension, Rademacher complexity).
- When and why a learning algorithm generalizes to unseen inputs — the bias-complexity tradeoff.
- The role of inductive bias — why two algorithms that both fit training data can differ wildly on out-of-distribution inputs.
- OOD generalization and distribution shift — the formal framework for why length generalization is hard.

Length generalization failure is a specific instance of out-of-distribution failure: the training distribution over n-digit numbers is consistent with infinitely many functions, most of which fail on (n+1)-digit inputs. The question is whether the architecture's inductive bias is strong enough to select the generalizing function from the consistent set — and the evidence is that it is not.

---

## Stage 5 — Transformers as Formal Objects
*Months 16–20*

By this stage you have the mathematical infrastructure. The goal is to understand transformers not as engineering systems but as formal mathematical objects whose expressibility can be characterized precisely.

Read in this order:

1. Vaswani et al. (2017) — *Attention Is All You Need.* Read formally. What are the algebraic operations? What is the function class defined by the architecture?

2. Weiss et al. (2021) — *Thinking Like Transformers* (the RASP paper). Implement RASP programs. Attempt to write RASP for multi-digit addition. Observe where it breaks and why.

3. Hahn, M. (2020) — *Theoretical Limitations of Self-Attention in Neural Sequence Models.* Your first formal lower bound paper in this domain.

4. Merrill, W. & Sabharwal, A. (2023) — *Saturated Transformers are Constant-Depth Threshold Circuits.* The bridge between transformer architecture and circuit complexity.

5. Bhattamishra, S. et al. (2020) — *On the Ability and Limitations of Transformers to Recognize Formal Languages.*

6. Pérez et al. (2021) — *Attention is Turing Complete.* Read carefully and identify which assumptions are required for Turing completeness, and whether they are empirically realistic.

---

## Stage 6 — The Empirical Literature
*Concurrent with Stage 5*

Read these papers not for their conclusions but to understand precisely what the experiments measure, where the theory fails to explain the data, and what questions remain open. The goal is to build a precise map of the gap between the formal lower bound results and the empirical phenomena.

### Core Empirical Papers

- Anil et al. (2022) — *Exploring Length Generalization in Large Language Models.* The most systematic empirical study. Read for its experimental design, not just its findings.
- Lee et al. (2023) — Format manipulation results: right-to-left representation, zero-padding. Why do these help? What does this reveal about the failure mode?
- Zhou et al. (2024) — Positional encoding ablations on arithmetic tasks.
- Nanda et al. (2023) — *Grokking and modular arithmetic.* Mechanistic interpretability on small arithmetic tasks. The most important empirical result for Sub-Problem 3.
- Wei et al. (2022) / Nye et al. (2021) — Chain-of-thought and scratchpad. Why does externalizing intermediate computation help? The formal answer connects to the depth-vs-parallelism question.

---

# Part III — How to Study This

## On Depth vs. Breadth

Go deep on Sipser and on Arora-Barak Chapters 1–6 and 14. Everything else can be absorbed at 70% depth because you will return to it with specific questions once you are inside the research. But circuit complexity lower bounds — understand the Håstad switching lemma at the proof level. It is the hardest material in the curriculum and the most important. The temptation to understand it conceptually without following the proof in detail will cost you significantly later.

## On Problem-Solving Practice

Do exercises. Not all of them — but the starred and hard problems in Sipser, and the exercises in Arora-Barak that you cannot immediately solve. The reason is specific: this research requires constructing lower bound arguments, and that is a skill acquired by practicing lower bound construction in simpler contexts. No amount of reading replaces the construction practice.

## On the Unifying Thread

The deepest conceptual thread in this curriculum runs as follows:

**group theory → semigroup theory → syntactic monoids → star-free languages → aperiodic automata → what transformers can compute**

Most researchers never see this thread because they studied algebra and theoretical computer science in separate courses with no connecting narrative. Keep it explicit at every stage. When you are studying group homomorphisms in Dummit & Foote, know that you are building toward the syntactic monoid characterization of regular languages. When you are studying AC⁰ lower bounds, know that you are building toward the formal expressibility ceiling of transformer attention.

## On Time

Realistically: 18–24 months of serious part-time work, or 12–18 months of full immersion. The bottleneck is not breadth — it is that circuit complexity lower bounds and formal language theory take time to genuinely internalize. You will believe you understand Håstad and then realize two months later that you did not. This is normal and expected.

## On Where to Begin

If you started from zero today: spend the first six months doing almost nothing that looks like machine learning research. You would be reading Velleman, then Abbott, then Sipser, doing exercises. That is correct. The field suffers from researchers who skipped this foundation. The researchers doing the best theoretical work in this space are the ones who own the automata theory and the circuit complexity, not the ones who own the deep learning engineering.

---

# Appendix — Complete Reading List

## Foundational Mathematics

- Velleman, D.J. — *How to Prove It* (3rd ed.), Cambridge University Press
- Hammack, R. — *Book of Proof* (3rd ed.), free at bookofproof.org
- Axler, S. — *Linear Algebra Done Right* (3rd ed.), Springer
- Abbott, S. — *Understanding Analysis* (2nd ed.), Springer
- Folland, G.B. — *Real Analysis: Modern Techniques and Their Applications* (2nd ed.), Wiley
- Dummit, D.S. & Foote, R.M. — *Abstract Algebra* (3rd ed.), Wiley

## Theoretical Computer Science

- Sipser, M. — *Introduction to the Theory of Computation* (3rd ed.), Cengage
- Arora, S. & Barak, B. — *Computational Complexity: A Modern Approach*, Cambridge University Press
- Hopcroft, J., Motwani, R., & Ullman, J. — *Introduction to Automata Theory, Languages, and Computation* (3rd ed.), Pearson
- Kushilevitz, E. & Nisan, N. — *Communication Complexity*, Cambridge University Press

## Information Theory and Probability

- Cover, T.M. & Thomas, J.A. — *Elements of Information Theory* (2nd ed.), Wiley
- Durrett, R. — *Probability: Theory and Examples* (5th ed.), Cambridge University Press

## Statistical Learning Theory

- Shalev-Shwartz, S. & Ben-David, S. — *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press

## Research Papers — Formal Theory

- Hahn, M. (2020). Theoretical Limitations of Self-Attention in Neural Sequence Models. *Transactions of the ACL.*
- Merrill, W. & Sabharwal, A. (2023). Saturated Transformers are Constant-Depth Threshold Circuits. *TACL.*
- Weiss, G., Goldberg, Y., & Yahav, E. (2021). Thinking Like Transformers. *ICML.*
- Bhattamishra, S., Ahuja, K., & Goyal, N. (2020). On the Ability and Limitations of Transformers to Recognize Formal Languages. *EMNLP.*
- Pérez, J., Barceló, P., & Marinkovic, J. (2021). Attention is Turing Complete. *JMLR.*
- Schützenberger, M.P. (1965). On Finite Monoids Having Only Trivial Subgroups. *Information and Control.*

## Research Papers — Empirical

- Anil, C. et al. (2022). Exploring Length Generalization in Large Language Models. *NeurIPS.*
- Lee, N. et al. (2023). Teaching Arithmetic to Small Transformers. *arXiv.*
- Nanda, N. et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability. *ICLR.*
- Zhou, H. et al. (2024). What Algorithms Can Transformers Learn? A Study in Length Generalization. *ICLR.*
- Nye, M. et al. (2021). Show Your Work: Scratchpads for Intermediate Computation. *arXiv.*
- Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS.*