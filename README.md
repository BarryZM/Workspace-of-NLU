# Workspace of Nature Language Understanding

## Target
+ 复现当前主流的文本分类方法
+ 高效美观的代码实现

   
## File System 
+ data
+ output
+ script_train
    + train model
+ script_pb_predict
    + load pb model
    + single case predict
    + batch predict
+ script_ensemble
    + boosting???
    + voting
    + blending
    + stacking
+ model
    + TextCNN and varients
    + BiRNN adn varients
    + fasttext
    + capsule
    + transformer
    + bert
    + meta learner for stacking (Logistic Regression)
+ utils
    + build_model.py
    + data_helper.py
    + tokenization.py(use in bert)
    + ...

## [Summmary of NLU](https://github.com/Apollo2Mars/Algorithms-of-Artificial-Intelligence/blob/master/9-Nature-Language-Processing/8-Dialog-System/2-NLU/README.md)
    
## Next
+ 代码逐行review，优化代码的逻辑性和简洁性
+ 工程优化
    + 定义命名规范
    + parser 和 flag 使用方式要统一
    + parser 变量名规范化（有的文件的parser 使用的有问题）
    + train dev test 的运行时间逻辑有问题
    + tensorboard
    + """检测文件是否存在，如果存在，则不执行此函数"""
    + 外层代码全部转化为 jupyter notebook
+ 预处理
    + embedding
        + 重新训练（未处理）
    + 字典
        + 当前使用的是tencent 的字典
        + 生成的字典暂时并未使用
        + cnn/rnn 的字典和其他模型的尚未统一
        + 字典重新生成
+ 模型
    + capsule
    + DPCNN
    + https://www.cnblogs.com/jiangxinyang/p/10210813.html
    + Attention-Based CNN 
    + cnn 变体
    + bert + DNNs
    
+ stacking-train-base-learner 多卡，多线程加速
+ 长文本和短文本的SOTA 方法整理
+ contexted-based TEXT CLF 建模
+ 1.   基于domain，intent，slot和其他信息（知识库，缠绕词表，热度）的re-rank策略。 https://arxiv.org/pdf/1804.08064.pdf
+ 2.   Joint-learning或multi-task策略，辅助未达标的分类领域。 https://arxiv.org/pdf/1801.05149.pdf
+ 3.   利用Bert embedding 进行再训练，如Bert embedding + Bi-LSTM-CRF。https://github.com/BrikerMan/Kashgari


