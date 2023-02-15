---
layout: post
title: "How Good is ChatGPT at Logical Reasoning?"
date: 2023-02-14 
---
*In this post, I have employed open-domain test cases to explore ChatGPT's proficiency in logical reasoning. The examples presented have demonstrated unequivocally that ChatGPT possesses remarkable logical reasoning skills, enabling it to solve a diverse set of open-domain reasoning tasks with outstanding performance.*
<br><br>

Logical reasoning is a fundamental aspect of human intelligence, but for a long time, computers have been unable to handle even basic reasoning tasks. Despite significant AI research efforts in recent decades, it has been widely believed that computers are incapable of performing meaningful logical reasoning in open-domain tasks. However, in a groundbreaking development, *ChatGPT* has demonstrated an astonishingly superior ability to perform logical reasoning in a wide range of unconstrained tasks, approaching or even matching human performance in many complex reasoning tasks.

Logical reasoning is a cognitive process by which a conclusion is drawn from a set of given conditions. It typically involves making inferences by starting with a set of premises and deriving conclusions based on those premises. The ultimate goal of logical reasoning is to construct valid arguments that would be deemed convincing by any rational individual.
As a core component of human intelligence $[1]$, logical reasoning plays a crucial role in selecting and interpreting information within a given context, establishing connections between concepts, and verifying and drawing conclusions. Logical reasoning can be broadly categorized into two main types: *deductive reasoning* and *non-deductive reasoning*. Deductive reasoning employs rigorous rules, such as logic or arithmetical calculations, to provide the strongest form of support, where the premises ensure the conclusion, and the conclusion is guaranteed to be true if all the premises are true. Conversely, non-deductive reasoning relies on premises that rationally support the conclusion without guaranteeing its truth. This is often understood in terms of probability, where the premises increase the likelihood that the conclusion is true and strong inferences make it highly likely. Non-deductive reasoning is a central feature of everyday life and most scientific fields, with important types including inductive, abductive, and analogical reasoning $[2]$.
In logical reasoning, some arguments are based solely on the given premises, while others involve several implicitly assumed premises that are considered obvious to most individuals. These implicit premises are commonly referred to as "commonsense". For the purposes of this discussion, our focus will be on the former cases where little to no commonsense is involved in the reasoning process. A follow-up post will explore *ChatGPT*'s performance in handling commonsense.

Traditionally, computers require manually programmed logic rules to perform logical reasoning, but it's challenging to specify a complete set of rules even for basic tasks, let alone open-domain tasks. This leads to brittle and error-prone reasoning when given conditions differ slightly from what was initially programmed. There has been a long-standing goal to let computers learn all necessary logic rules for reasoning, but little progress has been made until *ChatGPT*.

In contrast, *ChatGPT* is a remarkable AI system that has demonstrated real reasoning skills, including the ability to understand unconstrained situations described in natural language and draw logical conclusions convincingly. It can even handle complex arguments that require multiple steps of reasoning. Most impressively, *ChatGPT* can handle logical reasoning in any unconstrained situation, whether realistic or hypothetical, as long as it is reasonable to humans. This is what I refer to as "open-domain reasoning" in this post. To my knowledge, no other computer system has demonstrated such open-domain reasoning skills prior to *ChatGPT*.

Measuring *ChatGPT*'s logical reasoning skills is an interesting question, but evaluating it using common datasets designed to assess AI systems' reasoning abilities may not be suitable. Firstly, these datasets were created prior to *ChatGPT* and were widely available on the internet, which means *ChatGPT* may have incorporated them into its training process. Secondly, these datasets were designed to evaluate basic reasoning skills and may be too simple for *ChatGPT*, which can often approach these tasks from a broader and more complex perspective, producing answers that differ from the reference solutions provided, but are equally or even more effective to humans.

In this post, I will create new test cases to systematically evaluate *ChatGPT*'s logical reasoning abilities across different scenarios. I have developed these test cases from scratch to ensure that *ChatGPT* has not encountered them in its training data. These test cases are significantly more challenging and intricate than most available datasets.
Through these test cases, we will examine *ChatGPT*'s performance in various logical reasoning tasks, including deductive and non-deductive reasoning. Additionally, we will briefly discuss how *ChatGPT* acquired its remarkable reasoning skills, which were not manually programmed into its training process.
While I have designed and tested *ChatGPT* with dozens of test cases, in this post, I will report and discuss 10 representative ones. It is worth noting that the number of test cases in this study is relatively small. Therefore, readers are encouraged to create their own test cases to assess *ChatGPT*'s reasoning abilities in even more diverse situations.

### **ChatGPT on Deductive Reasoning**

Firstly, we will consider some complex scenarios to test *ChatGPT*'s deductive reasoning capabilities. In Example 1, we demonstrate that *ChatGPT* is able to accurately answer a complex factual question that has posed a challenge for previous Q&A systems. This question requires at least two reasoning steps: firstly, identifying all the relevant countries, and secondly, inferring the capitals of these countries. Although the question does not explicitly state these two steps, *ChatGPT* is capable of understanding what is implied and can successfully perform these two reasoning steps to arrive at the correct answer.

<figure align="center">
<figcaption> Example 1. A complex factual question
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-complex-factual-question.png" width="600" alt> 
</figure>

In Example 2, we demonstrate that *ChatGPT* is capable of completing a complex scheduling task. *ChatGPT* is able to fully comprehend all the conditions given in the question, infer the subway schedule from the given information, and then perform simple arithmetic calculations to determine the latest subway available to catch the meeting. Initially, *ChatGPT* provides a vague answer but is able to provide the precise information after a clarification question is posed.

<figure align="center">
<figcaption> Example 2. A fairly complex scheduling task
  </figcaption> 
  <img src="{{site.url}}/figures/logic-reasoning-morning-subway.png" width="600" alt> 
</figure>

Example 3 showcases how *ChatGPT* is capable of utilizing its logical reasoning abilities to solve a multi-step math problem. The question demands an understanding of what each provided number represents, as well as a step-by-step tracking of the given information. In order to arrive at the final answer, various arithmetic calculations are required.
Furthermore, in the second part of this example, *ChatGPT* is able to suggest solutions to address the capacity issue by considering a broader context than the original question itself. These suggestions are practical and logical.

<figure align="center">
 <figcaption>  Example 3. Complex aruguments requiring multi-step math calculations
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-school-bus.png" width="600"  alt> 
  <img src="{{site.url}}/figures/logic-reasoning-school-bus-over-capacilty.png" width="600" alt> 
</figure>

Example 4 provides an example of deductive reasoning using basic logic rules such as induction, syllogism, and contradiction. It is evident that *ChatGPT* has a complete understanding of these logic rules and is capable of skillfully applying them to a specific case in order to draw correct conclusions.

<figure align="center">
 <figcaption> Example 4. A deductive reasoning example on logic
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-inductive-logic-1.png" width="600"  alt> 
  <img src="{{site.url}}/figures/logic-reasoning-inductive-logic-2.png" width="600"  alt> 
</figure>

### **ChatGPT on Non-deductive Reasoning**

Moving on, let us explore several non-deductive reasoning examples. Example 5 presents a straightforward abductive reasoning problem that involves drawing an inference from an observation to a fact that explains the observation. The scenario in this example offers two possible causes ("raining" or "water pipe leaking") for an effect ("wet driveway"). *ChatGPT* clearly comprehends the cause-effect relationship presented in the given conditions, and can consistently deduce the most probable reason for each observation. It also understands the "explain-away" effect, where one potential cause can render other possible causes unlikely.

<figure align="center">
 <figcaption> Example 5. A simple abductive reasoning problem
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-wet-driveway-abductive.png" width="600"  alt> 
</figure>

In Example 6, an open-ended abductive reasoning question is presented, and it is demonstrated that *ChatGPT* can fully comprehend all the information provided and consider it in a broader context beyond the scope of the original question. In both parts of the example, *ChatGPT* generates a list of five plausible explanations to address the situation posed in the question. Many of the reasons provided by *ChatGPT* were not initially considered when the question was originally composed. However, in retrospect, they appear to be reasonable explanations.

<figure align="center">
 <figcaption> Example 6. An open-ended abductive problem
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-class-low-turnout-1.png" width="600"  alt> 
  <img src="{{site.url}}/figures/logic-reasoning-class-low-turnout-2.png" width="600"  alt> 
</figure>

Example 7 presents another open-ended question that pertains to a common scenario encountered in daily life. This type of question does not have a definitive solution, but rather necessitates the use of non-deductive reasoning to support one's stance. In this example, *ChatGPT* initially provides an ambiguous response, but after further prompting, it leans towards an answer that aligns with our common sense. In both instances, *ChatGPT* provides compelling reasons to support its conclusions.

<figure align="center">
 <figcaption> Example 7. An open-ended non-deductive problem
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-restaurants.png" width="600"  alt> 
</figure>

Example 8 presents a hypothetical question that may necessitate probabilistic reasoning. From the given information, *ChatGPT* is able to accurately deduce which situations are impossible and which events are more probable than others. This type of logical reasoning is a commonplace in human intelligence as a means of dealing with uncertainties in real-world scenarios. Typically, this type of reasoning does not necessitate precise mathematical calculations, but instead involves rough estimates based on probability.

<figure align="center">
 <figcaption> Example 8. A probabilistic inference problem
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-jar-ball.png" width="600"  alt> 
</figure>

In Example 9, a hypothetical scenario is presented which concerns how an individual might respond in a particular situation. By understanding how most people would react in such a case, *ChatGPT* is able to analogously infer that Mary will likely react in a similar manner.

<figure align="center">
 <figcaption> Example 9. An analogical reasoning example
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-inductive-wedding-ring.png" width="600"  alt>  
</figure>

In Example 10, it is demonstrated how *ChatGPT* can handle a complex argument that requires multiple steps of logical reasoning. The problem involves understanding the operations of a business, computing profits, and evaluating the viability of the business. In both parts of this example, *ChatGPT* effectively synthesizes all the given information and provides well-reasoned responses. Moreover, it displays the ability to think beyond the immediate question and offer useful suggestions in each case.

<figure align="center">
 <figcaption> Example 10. A complex argument example
  </figcaption>
  <img src="{{site.url}}/figures/logic-reasoning-inductive-conveniencestore-1.png" width="600"  alt>  
  <img src="{{site.url}}/figures/logic-reasoning-inductive-conveniencestore-2.png" width="600"  alt>  
</figure>

In conclusion, the examples presented above clearly demonstrate *ChatGPT*'s remarkable abilities in performing complex logical reasoning tasks in open domains. Prior to the advent of *ChatGPT*, it was unimaginable for computers to solve such unconstrained reasoning tasks. However, within such a short time frame, *ChatGPT* has successfully solved these tasks, displaying strong reasoning skills that are comparable to those of humans.

### **How Has ChatGPT Acquired Reasoning Skills?**

In this section, we will briefly discuss how *ChatGPT* has developed its remarkable logical reasoning abilities. Due to the enormous size of its underlying model (see my previous post [What is ChatGPT?](https://incml.github.io/2023/01/28/What-Is-ChatGPT.html)), it is reasonable to believe that *ChatGPT* can effortlessly store all information from its extensive training set. However, memorization alone is not sufficient for abstraction and reasoning skills.

There are at least three hypotheses that attempt to explain how *ChatGP*  has achieved its exceptional logical reasoning skills.

- **Rote Hypothesis:** Given that the training set of *ChatGPT* is so vast that it encompasses nearly all of the information accessible on the internet prior to 2021, it is probable that *ChatGPT* has been exposed to all of the above examples (or others that are similar enough) during its training. Subsequently, *ChatGPT* has likely memorized these examples to some extent within its enormous model. Thus, the observed logical reasoning abilities of *ChatGPT* can be attributed to its capacity for retrieving the most relevant case from its model.

- **Reshuffling Hypothesis:** It is unlikely that *ChatGPT* saw the above examples in their entirety during its training. Instead, *ChatGPT* has observed the necessary pieces of information from various contexts in the training set, and has memorized these pieces of information individually, without any abstraction. The logical reasoning displayed is simply the result of *ChatGPT* successfully retrieving all relevant pieces of information from the model based on certain text correlation statistics, and then reshuffling these pieces of information into a coherent order.

- **Emerging Hypothesis:** Although *ChatGPT* has been trained on a vast amount of unstructured text data, it has not been explicitly taught about concepts or relationships between entities. This suggests that, like human intelligence which arises from a complex cerebral cortex, *ChatGPT*'s ability to understand abstract concepts and engage in logical reasoning has emerged as a result of its increasing complexity.

Before the advent of *ChatGPT*, it was commonly believed that the best that a large language model (LLM) could do was to learn how to recombine separate pieces of information from its training data into a seemingly reasonable argument, as the reshuffling hypothesis suggests. There was no evidence to suggest that large language models could do anything beyond this, such as abstracting concepts or making inferences beyond observed cases. However, the fact that *ChatGPT* can accurately identify and consistently refer to the correct entities even when the entities are modified or changed within a given question implies that it has developed the ability to abstract concepts from concrete examples.  Its superior logical reasoning skills across a range of tasks further indicate that *ChatGPT* has likely established a solid and coherent framework for these concepts, similar to the way humans do. Based on the limited evidence available, it appears that the emerging hypothesis is more favorable compared to the other two hypotheses.

The exact mechanism behind *ChatGPT*'s logical reasoning skills is still largely unknown. Empirical results suggest that these skills emerge only when the complexity and size of the underlying model surpass a certain threshold, implying that model complexity is a crucial factor for intelligence. This aligns with what has been observed in biological brains. For example, the large language model GPT-3, the basis for *ChatGPT*, has approximately 10,000 relatively independent and structurally homogeneous transformer heads, each containing 6,000,000 parameters $[3]$. An interesting comparison can be made to the human neocortex, which is composed of about 150,000 *cortical columns*, each consisting of roughly 100,000 neurons $[4]$. Like the transformer heads in GPT-3, the cortical columns in the human neocortex are uniform in structure and can function relatively independently. Human intelligence arises only when the number of cortical columns reaches a certain point, as the human neocortex significantly outgrows those of other mammals, despite being structurally alike.

At last, it has also been suggested that *ChatGPT* has primarily acquired its understanding of abstract concepts and logical reasoning through exposure to a vast array of computer programs during its training. Compared to natural language texts, computer programs have a clear logical flow that may have aided *ChatGPT* in learning how to infer conclusions from conditions. This speculation can be easily tested by retraining the large language model without using any computer programs. Even if this speculation proves to be true, there is still a noticeable difference between computer programs and the logical reasoning examples discussed above. These findings bring up intriguing questions for further exploration.

### **References**

$[1]$  Howard E Gardner, *Frames of Mind: The Theory of Multiple Intelligences*, 2011.

$[2]$  Logical Reasoning, Retrieved February 10, 2023, from [https://en.wikipedia.org/wiki/Logical_reasoning](https://en.wikipedia.org/wiki/Logical_reasoning).

$[3]$ Tom Brown, Benjamin Mann, *et. al.*, *Language Models are Few-Shot Learners*, 
arXiv, 2020, from [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165).

$[4]$ Jeff Hawkins, *A Thousand Brains: A New Theory of Intelligence*, 2021.
