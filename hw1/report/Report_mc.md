# 实验报告
## Tokenizer

### 算法实现和训练
1. 简述 BPE 算法：Byte Pair Encoding, 基于统计方法合并最常见的字符或子词对，逐步构造更紧凑的词表。具体过程为 1.将单词拆分成字节级单位（也是一个字符）；2.统计字符对出现频率；3.合并出现频率最高的字符对，得到新字符；4.将合并产生的新字符加入词表；5.循环以上过程直到词表的大小达到指定大小。
2. 训练 Tokenizer 的流程：训练 Tokenizer 实际上是为了得到一个字典，字典的 key 为
   1. 将输入的文本以字节为单位进行切分，创建初始词汇表，由于每个字节只能表示 0~255，所以词汇表的大小不超过256。
   2. 统计每个字符对出现的频率。
   3. 将出现频率最大的字符对组成一个新的字符，将这个字符分配一个新的编号（在简单的BPE算法中从256开始），然后将训练文本中的所有该字符对替换为生成的字符。
   4. 重复多次 merge 和 replace 操作直到词表达到上限。
   依据以上步骤得到的字典是一个双射，在之后 encode 操作中，仅仅需要根据字节查询编号（优先选择可以合并的字节对）；在 decode 过程中只需要根据 Token 的编号查询到对应的字节。
3. 训练tokenizer：训练数据在本地，调用 train 函数，得到 vocab 字典。
4. 对比 encode 再 decode 后的 manual.txt:
5. 对比 huggingface transformers中的tokenizer 和自己训练得到的 tokenizer:
   1. 中文：
   2. 英文

### 问题

1. ``ord()`` 查看字符的 Unicode， ``chr()`` 查看 Unicode 对应的单词。ord('北')=21271；ord('大')=22823；chr(22823)=大；chr(27169)=模；chr(22411)=型。
2. vocab size 大小的影响：size 大时，更多的字符被编码为一个 Token，这样可以减小 encoded 文本的长度，有助于处理长文本同时加速推理。但是弊端为需要足够多的训练数据，储存 vocab 需要较多的空间。size 小时，适用于形式复杂的语言（韩语），但是弊端为失去语义的完整性，Transformer 处理更长的输入消耗更多的时间。
3. 不能处理非常简单的字符串操作任务的原因：可能是这个字符串中的多个字符被 encode 为一个 token, LLm 只能看到这个token，看不到具体的细节。例如翻转任务，如果prompt修改为先逐个打印字符串中的字符，然后用相反的顺序打印就可能正确实现字符串翻转，
4. 非英语语言（例如日语）上表现较差的原因：首先与模型的训练数据有一定关系，现有的许多LLM 使用大量英语语料训练。此外，也与 tokenizer 的训练有关：相比于英语中的单词，韩语中的“chunk” 被 tokenizer 被分割为多个 Token, ，所以对同样语义的句子，韩语被 encode 后的 token 数量更多，增加了模型的负担。更多的 token 在限制 max-token context 的情况下，用尽了资源，而Transformer 仅能支持有限的 context Line。
5. LLM 在简单算术问题上表现不好：大于3位的数字会被 merge 为一个 Token，LLM 不能直接接受到精确的数。
6. 在编写 Python 代码时遇到比预期更多的困难：有的 Tokenizer （gpt2）将每一个空格作为一个 token，python 中有过多的 space，浪费上下文的空间。为了提高 coding 能力，希望输入更多的上下文，所以一些 Tokenizer 会压缩空格，但是输入的缩进信息会损失，这是 python 中的重要信息。
7. 遇到字符串 “<|endoftext|>” 时突然中断的原因：``<|endoftext|>`` 是 tokenizer 中的 Special tokens ，代表序列的终结。 ``<|endoftext|>`` encode 后的 Token使大模型认为输入终止。
8. 关于 “SolidGoldMagikarp” 的问题时 LLM 会崩溃：如 SolidgoldMagikarp 等词在在训练集中从未出现，相应的 Neuron 从未激活，在初始化阶段是任意的，从未更新过。所以输入后会产生错误输出。
9. 使用 LLM 时应该更倾向于使用 YAML 而不是 JSON：``YAML`` 格式文件的储存密度比 JSON 高。
10. 不是端到端的原因：LLM 处理的是 Token 输出的也是 Token，这些 Token 都需要 Tokenizer 来处理成为与人来交互的文本。